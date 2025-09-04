from argparse import ArgumentParser
import logging
from shutil import rmtree

import numpy as np
from scipy import signal

from joblib import cpu_count, delayed, Parallel
from pyrocko import io as pio, trace
from pyrocko.io import stationxml
from pyrocko.trace import InfiniteResponse, TraceTooShort
from pyrocko.util import UnavailableDecimation, ensuredir
import xarray as xr

from classes import FeatureExtractConfig, DummyEvent
from log_util import custom_logger, set_loglevel


logging.getLogger("pyrocko.io.stationxml").setLevel(logging.ERROR)
logger = custom_logger("extract_features")


def extend_extract(tr, tmin, tmax):
    """
    This function has been borrowed from Grond software.
    """
    deltat = tr.deltat
    itmin_frame = int(np.floor(tmin / deltat))
    itmax_frame = int(np.ceil(tmax / deltat))
    nframe = itmax_frame - itmin_frame + 1
    n = tr.data_len()
    a = np.empty(nframe, dtype=np.float64)
    itmin_tr = int(round(tr.tmin / deltat))
    itmax_tr = itmin_tr + n
    icut1 = min(max(0, itmin_tr - itmin_frame), nframe)
    icut2 = min(max(0, itmax_tr - itmin_frame), nframe)
    icut1_tr = min(max(0, icut1 + itmin_frame - itmin_tr), n)
    icut2_tr = min(max(0, icut2 + itmin_frame - itmin_tr), n)
    a[:icut1] = tr.ydata[0]
    a[icut1:icut2] = tr.ydata[icut1_tr:icut2_tr]
    a[icut2:] = tr.ydata[-1]
    tr = tr.copy(data=False)
    tr.tmin = itmin_frame * deltat
    tr.set_ydata(a)
    return tr


def preprocess(tr_raw, resp, preprocess_config):
    """
    Signal processing.
    Steps are: downsampling -> bandpass filtering -> tapering.

    Parameters
    ----------
    tr_raw : `pyrocko.trace.Trace` object
        Raw seismic trace to be processed.
    resp: `pyrocko.trace.FrequencyResponse` object
        Instrument response function.
    preprocess_config : `~PreprocessConfig` object
        Configuration that provides preprocessing parameters.

    Returns
    -------
    tr_proc : `pyrocko.trace.Trace` object
        Processed seismic trace. Processing steps are not applied
        inplace, but a new trace object is returned.

    Raises
    ------
    `pyrocko.util.UnavailableDecimation`
        If target sampling rate is above the sampling rate of `tr_raw`.
    `pyrocko.trace.TraceTooshort`, `pyrocko.trace.InfiniteResponse`
        If instrument response deconvolution fails.
    """
    config = preprocess_config
    if config.target_fs > (fs := 1.0 / tr_raw.deltat):
        raise UnavailableDecimation(
            f"{'.'.join(tr_raw.nslc_id)}: Target fs "
            f"({config.target_fs} Hz) is greater than "
            f"original fs ({fs} Hz)"
        )

    # Station response removal (including pre-filtering)
    tr_proc = tr_raw.transfer(
        tfade=config.tfade_taper,
        freqlimits=config.pre_filt,
        transfer_function=resp,
        cut_off_fading=False,
        invert=True,
    )

    # Downsampling
    tr_proc.downsample_to(config.target_deltat, snap=True, allow_upsample_max=5)
    # Bandpass filtering
    tr_proc.bandpass(4, config.fmin, config.fmax)
    # Tapering
    taperer = trace.CosTaper(
        tr_proc.tmin,
        tr_proc.tmin + config.tfade_taper,
        tr_proc.tmax - config.tfade_taper,
        tr_proc.tmax,
    )
    tr_proc = extend_extract(tr_proc, *taperer.time_span())
    tr_proc.taper(taperer, inplace=True, chop=False)
    return tr_proc


def compute_specgram(tr, specgram_config):
    """
    Compute spectrogram of a trace with consecutive Fourier transforms.
    This function uses `scipy.signal.spectrogram`.

    Parameters
    ----------
    tr : `pyrocko.trace.Trace` object
        Seismic trace.
    specgram_config : `~SpecgramConfig` object
        Configuration that provides parameters for computing spectrogram.

    Returns
    -------
    out: `xarray.DataArray` object, shape: (n_freqs, n_times)
      Spectrogram of `tr`. Data dimension names will be `dim_0="freqs"`
      and `dim_1="times"`, respectively.
    """
    config = specgram_config
    fs = 1.0 / tr.deltat

    # Number of samples in each window to take FFT
    nperseg = int(config.window_tlen * fs) + 1
    # Number of samples to overlap between windows
    noverlap = int(config.overlap * nperseg)
    # Kaiser window function (β=πα)
    window = signal.windows.kaiser(nperseg, beta=config.kaiser_beta)
    # To perform zero-padded FFT
    nfft = config.nfft or trace.nextpow2(nperseg * 1.2)

    freqs, times, specgram = signal.spectrogram(
        tr.ydata.astype(np.float64),
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )

    return xr.DataArray(
        data=specgram,
        coords={"freqs": freqs, "times": times + tr.tmin},  # actual times
        attrs={"deltat": (nperseg - noverlap) * tr.deltat, "nslc_id": tr.nslc_id},
    )


class NotEnoughData(Exception):
    """
    This exception is raised when not enough data is available.
    """

    pass


def compute_crop_specgram(tr_proc, tp, specgram_config):
    """
    This function first computes spectrogram of processed trace using
    `compute_specgram()` function above, then center the spectrogram on
    its spectral peak and crop it to given frequency range and time length.

    Parameters
    ----------
    tr_proc : `pyrocko.trace.Trace` object
        Processed trace. See `preprocess()` function above.
    tp: float
        P-wave arrival time (timestamp).
    specgram_config : `~SpecgramConfig` object
        Configuration that provides parameters for computing spectrogram.

    Returns
    -------
    out: `xarray.DataArray` object, shape: (n_freqs_crop, n_times_crop)
      Cropped spectrogram of `tr_proc`. Data dimension names will be
      `dim_0="freqs"` and `dim_1="times"`, respectively.

    Raises
    ------
    `pyrocko.trace.TraceTooShort`
        If the trace is too short to crop the spectrogram on time axis.
    `~NotEnoughData`
        If the trace is short on one side after finding time indices.
    """
    config = specgram_config

    if ((tbeg := tp - config.before_tp) < tr_proc.tmin) or (
        (tend := tp + config.after_tp) > tr_proc.tmax
    ):
        raise TraceTooShort(f"{'.'.join(tr_proc.nslc_id)}")

    # Chop trace to make specgram.times consistent between events
    if config.cut_first:
        tr_proc.chop(tbeg - 0.5 * config.tlen, tend + 0.5 * config.tlen)
    # Full spectrogram (DataArray)
    specgram = compute_specgram(tr_proc, config)

    # Number of time samples to crop
    n_tcrop = int(np.floor(config.tlen / specgram.deltat)) + 1

    # Find closest sample time to tp (times are actual trace times)
    i_anchor = np.abs(specgram.times.data - tp).argmin()
    t_anchor = specgram.times.data[i_anchor]
    i_tstart = np.argmax(specgram.times.data >= (t_anchor - config.before_tp))
    i_tstop = i_tstart + n_tcrop

    if i_tstop > specgram.times.size:
        raise NotEnoughData(
            f"{'.'.join(tr_proc.nslc_id)}: exceeded trace end "
            f"({i_tstop=} > {specgram.times.size=})"
        )

    fmask = np.logical_or(
        specgram.freqs.data < config.fmin, specgram.freqs.data > config.fmax
    )

    return xr.DataArray(
        data=specgram.data[~fmask, i_tstart:i_tstop],
        coords={
            "freqs": specgram.freqs.data[~fmask],
            "times": specgram.times.data[i_tstart:i_tstop],
        },
        attrs=specgram.attrs,
    )


def pipeline(tr_raw, resp, tp, pipeline_config):
    """
    Parameters
    ----------
    tr_raw : `pyrocko.trace.Trace` object
        Raw trace to be processed.
    resp: `pyrocko.trace.FrequencyResponse` object
        Instrument response function.
    tp: float
        P-wave arrival time (timestamp).
    pipeline_config : `~FeatureExtractPipelineConfig` object
        Configuration that provides full pipeline parameters.

    Returns
    -------
    `xarray.DataArray` if all processing steps performs completely.
    Data dimension names will be `dim_0="freqs"` and `dim_1="times"`,
    respectively.

    Raises
    ------
    See :py:func:`~preprocess` and :py:func:`~compute_crop_specgram`
    """
    # Signal processing
    tr_proc = preprocess(tr_raw, resp, pipeline_config.preprocess_config)
    # Cropped spectrogram
    return compute_crop_specgram(tr_proc, tp, pipeline_config.specgram_config)


def load_sort_traces(event, dataset_config):
    """
    Parameters
    ----------
    event : `pyrocko.model.Event` object
        Seismic event.
    dataset_config : `~FeatureExtractDatasetConfig`
        Configuration that provides parameters to access data.

    Returns
    -------
    trs : list of `pyrocko.trace.Trace` objects
        Seismic traces sorted by distance.
    """
    config = dataset_config

    # Loading and degapping traces
    mseed_fname = config.expand_mseed_fname_template(event)
    mseed_fpath = config.mseed_dirpath.joinpath(mseed_fname)
    trs = trace.degapper(pio.load(mseed_fpath.as_posix()), maxgap=21)
    if len(trs) == 0:
        # Empty
        logger.warning(f"Empty traces file: {mseed_fpath}")
        return trs

    # Computing event-station distances
    nsl_to_distance = dict.fromkeys(config.nsl_to_station.keys(), None)
    for tr in trs:
        nsl = tr.nslc_id[:-1]
        try:
            station = config.nsl_to_station[nsl]
            nsl_to_distance[nsl] = station.distance_to(event)
        except KeyError:
            continue

    # Sorting by distance while pushing None values to the end
    def f(tr):
        """
        This function returns either (True, None) or (False, distance).
        Since tuples are sorted item by item, all non-None elements will
        come first (because `False < True`).
        """
        distance = nsl_to_distance[tr.nslc_id[:-1]]
        return (distance is None, distance)

    trs.sort(key=f)

    return trs


def load_map_arrivals(event, dataset_config):
    """
    Parameters
    ----------
    event : `pyrocko.model.Event` object
        Seismic event.
    dataset_config : `~FeatureExtractDatasetConfig`
        Configuration that provides parameters to access data.

    Returns
    -------
    Dictionary that maps (net, sta) tuples to corresponding
    `~DummyArrival` objects.
    """
    arrivals_fname = dataset_config.expand_arrivals_fname_template(event)
    arrivals_fpath = dataset_config.arrivals_dirpath.joinpath(arrivals_fname)
    dummy_event = DummyEvent.load(filename=arrivals_fpath.as_posix())
    ns_to_arrival = dict()
    for arrival in dummy_event.arrival_list:
        ns_to_arrival[arrival.network_code, arrival.station_code] = arrival
    return ns_to_arrival


def process_one_event(args):
    event, config = args
    event_id = config.dataset_config.expand_mseed_fname_template(event)

    trs = load_sort_traces(event, config.dataset_config)  # sorted by dist
    ns_to_arrival = load_map_arrivals(event, config.dataset_config)

    for tr_raw in trs:
        try:
            resp = config.dataset_config.xmlresps.get_pyrocko_response(
                tr_raw.nslc_id, time=event.time, fake_input_units="M/S"
            )
        except stationxml.NoResponseInformation:
            logger.warning(f"No station response found: {'.'.join(tr_raw.nslc_id)}")
            continue

        try:
            tp = ns_to_arrival[tr_raw.network, tr_raw.station].time
            x = pipeline(tr_raw, resp, tp, config.pipeline_config)
        except (
            KeyError,
            UnavailableDecimation,
            TraceTooShort,
            InfiniteResponse,
            NotEnoughData,
        ):
            logger.exception(f"Unexpected Error: {event_id=}")
            continue
        else:
            specgram_fname = config.dataset_config.expand_spcgram_fname_template(event)
            specgram_fpath = config.dataset_config.specgram_dirpath.joinpath(
                specgram_fname
            )
            # Make sure that out directory exists
            ensuredir(config.dataset_config.specgram_dirpath.as_posix())
            # Save into netCDF
            x.to_netcdf(path=specgram_fpath.as_posix(), mode="w")
            break


def main():
    parser = ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="number of processes running in parallel",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="overwrite existing out directory/files",
    )
    parser.add_argument(
        "--loglevel",
        metavar="LOGLEVEL",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help='set logger level to "critical", "error", "warning", "info", '
        'or "debug" (default: info).',
    )

    args = parser.parse_args()

    set_loglevel(logger, args.loglevel)

    # Reading config file
    config = FeatureExtractConfig.read_from_file(args.config_file)
    # Check if all input files/dirs exist
    config.dataset_config.check_inpaths()
    # Check if out directory exists + further actions
    try:
        config.dataset_config.check_outpath()
    except FileExistsError as error:
        if args.force:
            logger.warning(
                "Out directory already exists and you are going to delete it."
            )
            prompt = input("Do you want to continue? [Y/n] ")
            if prompt.lower() not in ("y", "yes"):
                return
            else:
                rmtree(config.dataset_config.specgram_dirpath.as_posix())
        else:
            raise FileExistsError(
                f"{error=}.\n\nTo suppress this error, use --force option.\n"
            )

    # Check signal-processing parameters
    config.pipeline_config.preprocess_config.check_nyquist()

    # Check number of jobs
    prompt = None
    if args.num_jobs == 1:
        logger.warning("The process will be running on single CPU.")
        prompt = input("Do you want to continue? [Y/n] ")
    elif args.num_jobs == -1:
        logger.warning(f"You are going to use all available {cpu_count()} CPUs.")
        prompt = input("Do you want to continue? [Y/n] ")

    if prompt and prompt.lower() not in ("y", "yes"):
        return

    # Go and run the jobs!
    with Parallel(n_jobs=args.num_jobs, verbose=10) as parallel:
        _ = parallel(
            delayed(process_one_event)((event, config))
            for event in config.dataset_config.events
        )


if __name__ == "__main__":
    main()
