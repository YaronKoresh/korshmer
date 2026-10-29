import pytest

from korshmer.network.ndi_output import NdiOutput
from korshmer.network.youtube_live import YouTubeLiveStreamer

NAN = float("nan")
INF = float("inf")

BAD_RATES = [NAN, INF, "48000", None, True, 44100.5, -1, 0, 7999, 192001]
BAD_CHANNELS = [True, 1.5, "2", None, NAN, 0, 3]
BAD_QUEUES = [0, -5, "64", None, NAN, 1.5, True, 10**7]


@pytest.mark.parametrize("rate", BAD_RATES)
def test_youtube_sample_rate_must_be_a_supported_whole_number(rate):
    with pytest.raises(ValueError, match="sample_rate"):
        YouTubeLiveStreamer("key-1", sample_rate=rate)


@pytest.mark.parametrize("channels", BAD_CHANNELS)
def test_youtube_channels_must_be_one_or_two(channels):
    with pytest.raises(ValueError, match="channels"):
        YouTubeLiveStreamer("key-1", channels=channels)


@pytest.mark.parametrize("size", BAD_QUEUES)
def test_youtube_queue_size_must_be_a_positive_whole_number(size):
    with pytest.raises(ValueError, match="max_queue"):
        YouTubeLiveStreamer("key-1", max_queue=size)


@pytest.mark.parametrize("ratio", [NAN, INF, -1, 0, 1.5, "0.8", None, True])
def test_youtube_congestion_ratio_must_be_above_zero_and_at_most_one(ratio):
    with pytest.raises(ValueError, match="congestion_ratio"):
        YouTubeLiveStreamer("key-1", congestion_ratio=ratio)


@pytest.mark.parametrize(
    "url",
    ["http://x", "", None, "rtmp://", "rtmp://a b/live\nfoo", 5, "rtmp://h?x"],
)
def test_youtube_ingest_url_must_be_an_rtmp_address(url):
    with pytest.raises(ValueError, match="ingest_url"):
        YouTubeLiveStreamer("key-1", ingest_url=url)


def test_youtube_valid_settings_are_kept():
    s = YouTubeLiveStreamer(
        "key-1",
        sample_rate=44100,
        channels=1,
        max_queue=8,
        congestion_ratio=0.5,
        ingest_url="rtmps://live.example.org:443/app",
    )
    assert (s.sample_rate, s.channels) == (44100, 1)
    assert s._congested_at == 4


@pytest.mark.parametrize("rate", BAD_RATES)
def test_ndi_sample_rate_must_be_a_supported_whole_number(rate):
    with pytest.raises(ValueError, match="sample_rate"):
        NdiOutput(sample_rate=rate)


@pytest.mark.parametrize("channels", BAD_CHANNELS)
def test_ndi_channels_must_be_one_or_two(channels):
    with pytest.raises(ValueError, match="channels"):
        NdiOutput(channels=channels)


@pytest.mark.parametrize("size", BAD_QUEUES)
def test_ndi_queue_size_must_be_a_positive_whole_number(size):
    with pytest.raises(ValueError, match="max_queue"):
        NdiOutput(max_queue=size)
