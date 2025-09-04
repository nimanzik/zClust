from functools import lru_cache
import logging
import math
from pathlib import Path
from string import Template

from pyrocko import model as pmodel
from pyrocko.io import stationxml
from pyrocko.guts import (
    Any,
    Bool,
    Choice,
    Dict,
    Float,
    Int,
    List,
    Object,
    String,
    StringChoice,
    Tuple,
)
from pyrocko.trace import AboveNyquist


guts_prefix = "pf"


logging.getLogger("pyrocko.io.stationxml").setLevel(logging.ERROR)


class PreprocessConfig(Object):
    target_fs = Float.T(help="Target sampling rate to downsample the trace. Unit: Hz")
    fmin = Float.T(help="Lower corner frequency of the bandpass filter. Unit: Hz")
    fmax = Float.T(help="Upper corner frequency of the bandpass filter. Unit: Hz")
    ford = Int.T(help="Order of the filter. Default: 4", default=4)
    tfade_taper = Float.T(
        help="Rise/fall time of cosine taper applied in time domain at "
        "both ends of a trace. Unit: s"
    )
    pre_filt = Tuple.T(
        n=4,
        content_t=Float.T(),
        help="4-tuple with corner frequencies of a cosine taper applied "
        "in frequency domain before response deconvolution. Unit: Hz",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_nyquist = self.target_fs / 2.0
        self.target_deltat = 1.0 / self.target_fs

    def check_nyquist(self):
        """
        Raises
        ------
        `pyrocko.trace.AboveNyquist`
            If either `fmin` or `fmax` or both are above the Nyquist
            frequency after decimation (`target_nyquist`).
        """
        freqs = [self.fmin, self.fmax]
        intros = ["Lower corner freq. of bandpass", "Higher corner freq. of bandpass"]

        messages = []
        for freq, intro in zip(freqs, intros):
            if freq >= self.target_nyquist:
                messages.append(
                    f"{intro} ({freq} Hz) is >= target Nyquist "
                    f"frequency ({self.target_nyquist} Hz)"
                )

        if messages:
            raise AboveNyquist("; ".join(messages))


class SpecgramConfig(Object):
    window_tlen = Float.T(help="Time length of each window to take FFT. Unit: s")
    overlap = Float.T(help="Overlap between windows. Must be between 0 and 1.")
    kaiser_beta = Float.T(
        help="Kaiser-window shape parameter. As beta gets large, "
        "the Kaiser window narrows. Default: 1.8π",
        default=1.8 * math.pi,
    )
    nfft = Int.T(
        help="Length of the FFT used (optional). If not given, the FFT "
        "length will be `nextpow2(nperseg * 1.2)`",
        optional=True,
    )
    before_tp = Float.T(
        help="Offset before P arrival to crop final spectrogram. Unit: s"
    )
    after_tp = Float.T(help="Offset after P arrival to crop final spectrogram. Unit: s")
    fmin = Float.T(help="Lower frequency bin to crop final spectrogram. Unit: Hz")
    fmax = Float.T(help="Upper frequency bin to crop final spectrogram. Unit: Hz")
    cut_first = Bool.T(
        help="If true (default), trace is cut before computing its "
        "spectrogram. Cutting time span is `tp - (before_tp + tlen/2)` "
        "and `tp + (after_tp + tlen/2)`, where "
        "`tlen = before_tp + after_tp`."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tlen = self.before_tp + self.after_tp


class FeatureExtractPipelineConfig(Object):
    preprocess_config = PreprocessConfig.T()
    specgram_config = SpecgramConfig.T()


class FeatureExtractDatasetConfig(Object):
    events_fname = String.T(
        help="Catalog of seismicity. Format: Pyrocko events", yamlstyle="'"
    )
    stations_fname = String.T(
        help="Stations information. Format: Pyrocko stations", yamlstyle="'"
    )
    xmlresps_fname = String.T(
        help="Station responses file in FDSNStationXML format.", yamlstyle="'"
    )
    mseed_dirname = String.T(
        help="Directory where miniSEED files are stored", yamlstyle="'"
    )
    mseed_fname = String.T(
        help="miniSEED file names template. Must include a *braced* "
        "placeholder of the form ${event__<identifier>}, where "
        '"identifier" is either Pyrocko event attribute "name" '
        "or a user-defined event attribute matching a dictionary "
        'in Pyrocko event attribute "extras".',
        yamlstyle="'",
    )
    arrivals_dirname = String.T(
        help="Directory where dummy events with dummy arrivals are stored",
        yamlstyle="'",
    )
    arrivals_fname = String.T(
        help="Dummy-event file names template. Must include a *braced* "
        "placeholder of the form ${event__<identifier>}, where "
        '"identifier" is either DummyEvent attribute "name" or '
        "a user-defined attribute matching a dictionary key in "
        'DummyEvent attribute "extras".',
        yamlstyle="'",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_fpath = Path(self.events_fname)
        self.stations_fpath = Path(self.stations_fname)
        self.xmlresps_fpath = Path(self.xmlresps_fname)

        self.mseed_dirpath = Path(self.mseed_dirname)
        self.mseed_fname_template = Template(self.mseed_fname)
        self.expand_mseed_fname_template = self._create_template_expander(
            self.mseed_fname_template
        )

        self.arrivals_dirpath = Path(self.arrivals_dirname)
        self.arrivals_fname_template = Template(self.arrivals_fname)
        self.expand_arrivals_fname_template = self._create_template_expander(
            self.arrivals_fname_template
        )

        self.specgram_dirpath = self.mseed_dirpath.parent.joinpath("prepared")
        self.specgram_fname_template = Template(
            f"${{{self._extract_template_placeholder()}}}.nc"
        )  # netCDF
        self.expand_spcgram_fname_template = self._create_template_expander(
            self.specgram_fname_template
        )

    def _extract_template_placeholder(self):
        """
        Extract *braced* placeholder in `mseed_fname_template` object.
        """
        st = self.mseed_fname_template
        return st.pattern.search(st.template).groupdict()["braced"]

    def _create_template_expander(self, st):
        """
        Create a callable object that takes `pyrocko.model.Event` object
        as operand and performs template substitution.
        """
        placeholder = self._extract_template_placeholder()
        attrib = placeholder.split("__")[1]
        if hasattr(pmodel.Event(), attrib):

            def expand_template(event):
                return st.substitute({placeholder: event.attrib})
        else:

            def expand_template(event):
                return st.substitute({placeholder: event.extras[attrib]})

        return expand_template

    @property
    @lru_cache
    def events(self):
        return pmodel.load_events(self.events_fpath.as_posix())

    @property
    def stations(self):
        return pmodel.load_stations(self.stations_fpath.as_posix())

    @property
    @lru_cache
    def xmlresps(self):
        return stationxml.load_xml(filename=self.xmlresps_fpath.as_posix())

    @property
    def nsl_to_station(self):
        """
        Returns
        -------
        A dictionary mapping `(net, sta, loc)` tuples to corresponding
        `pyrocko.model.Station` objects.
        """
        return dict((s.nsl(), s) for s in self.stations)

    def check_inpaths(self):
        """
        Raises
        ------
        FileNotFoundError
            If one or more of the following paths do not exist:
            `events_fpath`, `stations_fpath`, `mseed_dirpath`.
        """
        paths = (
            self.events_fpath,
            self.stations_fpath,
            self.xmlresps_fpath,
            self.mseed_dirpath,
            self.arrivals_dirpath,
        )
        intros = ["Events", "Stations", "XMLResps", "miniSEEDs", "Arrivals"]
        messages = []
        for path, intro in zip(paths, intros):
            if not path.exists():
                messages.append(f"{intro}: {path}")

        if messages:
            raise FileNotFoundError(f"no such path(s): {'; '.join(messages)}")

    def check_outpath(self):
        """
        Raises
        ------
        FileExistsError
            If out directory already exists.
        """
        if self.specgram_dirpath.is_dir():
            raise FileExistsError(
                f"out directory already exists: {self.specgram_dirpath}"
            )


class FeatureExtractConfig(Object):
    path_prefix = String.T(
        help="All files/directories referenced are treated relative "
        "to the location of the configuration file"
    )
    dataset_config = FeatureExtractDatasetConfig.T()
    pipeline_config = FeatureExtractPipelineConfig.T()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.basepath = None

    @classmethod
    def read_from_file(cls, filepath):
        """
        Parameters
        ----------
        filepath : str or `pathlib.Path` object
            Config file.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Loading config from file
        config = cls.load(filename=filepath.as_posix())
        # Setting config base path (contaning directory)
        config.basepath = filepath.parent.absolute().resolve()

        # Path relative to the `filepath`
        relpath = config.basepath.joinpath(config.path_prefix).resolve()
        for path_name in (
            "events_fpath",
            "stations_fpath",
            "xmlresps_fpath",
            "mseed_dirpath",
            "arrivals_dirpath",
            "specgram_dirpath",
        ):
            # Resoving data paths
            old = getattr(config.dataset_config, path_name)
            new = relpath.joinpath(old)
            setattr(config.dataset_config, path_name, new)

        return config


class DummyArrival(Object):
    network_code = String.T()
    station_code = String.T()
    phase = String.T()
    time = Float.T(help="pick.time.value + arrival.time_correction")
    time_residual = Float.T(optional=True)
    distance = Float.T(optional=True, help="Unit: deg")


class DummyEvent(Object):
    name = String.T(default="", optional=True, yamlstyle="'")
    arrival_list = List.T(DummyArrival.T(), default=[])
    extras = Dict.T(String.T(), Any.T(), default={})


class TrainDatasetConfig(Object):
    events_fname = String.T(
        help="Catalog of seismicity. Format: Pyrocko events", yamlstyle="'"
    )
    root_dirname = String.T(help="Directory with all netCDF arrays.", yamlstyle="'")
    sample_fname = String.T(
        help="Sample file name template. Must include a *braced* placeholder "
        "of the form ${sample_id}.",
        yamlstyle="'",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events_fpath = Path(self.events_fname)
        self.root_dirpath = Path(self.root_dirname)
        self.sample_fname_template = Template(self.sample_fname)

    @property
    @lru_cache
    def events(self):
        return pmodel.load_events(self.events_fpath.as_posix())

    @property
    @lru_cache
    def sample_ids(self):
        return [ev.extras["dummy_name"] for ev in self.events]


class CAETrainParamsConfig(Object):
    batch_size = Int.T(help="Input batch size for training")
    num_epochs = Int.T(help="Number of epochs to train")
    learning_rate = Float.T(help="Learning rate")


class CAEDECTrainParamsConfig(Object):
    batch_size = Int.T(help="Input batch size for training")
    num_epochs = Int.T(help="Number of epochs to train")
    learning_rate = Float.T(help="Learning rate")
    n_clusters = Int.T(help="Numebr of clusters to form")
    n_features = Int.T(help="The number of latent space dimensions.")
    tol = Float.T(help="Threshold for minimum cluster-assignment change")
    update_interval = Int.T(help="Update target distribution every k epochs")
    clust_loss_weight = Float.T(help="Parameters that balances two losses")
    clust_init_method = StringChoice.T(choices=["kmeans", "gmm"], yamlstyle="'")
    pretrained_model = String.T(
        help='Pretrained autoencoder. Must be the file "name" only. It will '
        'be automatically combined with "<run_dirname>/<problem_name>/" '
        "to get accessed.",
        yamlstyle="'",
    )


class TrainConfig(Object):
    path_prefix = String.T(
        help="All files/directories referenced are treated relative "
        "to the location of the configuration file",
        yamlstyle="'",
    )
    problem_name = String.T(
        help="Name used to identify the output directory.", yamlstyle="'"
    )
    run_dirname = String.T(
        help="Parent directory where to save the output directory.", yamlstyle="'"
    )
    dataset_config = TrainDatasetConfig.T()
    params_config = Choice.T(
        choices=[CAETrainParamsConfig.T(), CAEDECTrainParamsConfig.T()]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run_dirpath = Path(self.run_dirname)
        self.basepath = None

    @classmethod
    def read_from_file(cls, filepath):
        """
        Parameters
        ----------
        filepath : str or `pathlib.Path` object
            Config file.
        """
        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Loading config from file
        config = cls.load(filename=filepath.as_posix())
        # Setting config base path (contaning directory)
        config.basepath = filepath.parent.absolute().resolve()

        # Path relative to the `filepath`
        relpath = config.basepath.joinpath(config.path_prefix).resolve()
        for path_name in ("events_fpath", "root_dirpath"):
            # Resoving data paths
            old = getattr(config.dataset_config, path_name)
            new = relpath.joinpath(old)
            setattr(config.dataset_config, path_name, new)

        config.run_dirpath = relpath.joinpath(config.run_dirpath)
        config.save_dirpath = config.run_dirpath.joinpath(config.problem_name)

        return config


__all__ = """
    PreprocessConfig
    SpecgramConfig
    FeatureExtractPipelineConfig
    FeatureExtractDatasetConfig
    FeatureExtractConfig
    DummyArrival
    DummyEvent
    TrainDatasetConfig
    TrainParamsConfig
    TrainConfig
""".split()
