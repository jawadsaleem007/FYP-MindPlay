"""List available LSL streams (name, type, channel count, nominal sampling rate).

Run this before the real-time classifier to verify your Smarting EEG stream
is visible. Example:

    python scripts/list_lsl_streams.py

"""
from pylsl import resolve_streams


def main(timeout=2.0):
    streams = resolve_streams(timeout=timeout)
    if not streams:
        print("No streams found in", timeout, "seconds.")
        return
    print(f"Found {len(streams)} stream(s):")
    for i, s in enumerate(streams):
        info = s
        print(f"[{i}] Name='{info.name()}' Type='{info.type()}' Channels={info.channel_count()} SFreq={info.nominal_srate()} SourceId={info.source_id()}")


if __name__ == "__main__":
    main()
