
def get_channel_labels(info, debug=False):
    labels = []
    try:
        desc = info.desc()
        if debug:
            print(f"[DEBUG] Stream descriptor found")
        
        ch = desc.child('channels').child('channel')
        if debug:
            print(f"[DEBUG] First channel node: {ch}")
        
        count = 0
        while ch and ch.name():
            label = ch.child_value('label') or ''
            labels.append(label)
            if debug:
                print(f"[DEBUG]   Channel {count}: label='{label}'")
            ch = ch.next_sibling()
            count += 1
        
        if debug:
            print(f"[DEBUG] Total labels extracted: {len(labels)}")
    except Exception as e:
        if debug:
            print(f"[DEBUG] Exception parsing labels: {e}")
        pass
    
    if debug:
        print(f"[DEBUG] Channel count: {info.channel_count()}, Labels count: {len(labels)}")
    
    if len(labels) != info.channel_count():
        if debug:
            print(f"[DEBUG] Mismatch detected! Returning empty labels")
        labels = [''] * info.channel_count()
    
    return labels


def test_channel_picks():
    """Test to find labels and indices for specific channel picks."""
    from pylsl import StreamInlet, resolve_byprop
    
    print("Looking for EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=5.0)
    if not streams:
        print("No EEG stream found!")
        return
    
    stream_info = streams[0]
    inlet = StreamInlet(stream_info)
    
    labels = get_channel_labels(stream_info, debug=True)
    print(f"\nTotal channels: {stream_info.channel_count()}")
    print(f"All channel labels: {labels}\n")
    
    # Test picks
    test_channels = ['fp1', 'fp2', 'cz', 'c3', 'c4']
    print(f"Looking up channels: {test_channels}")
    
    lower_map = {lab.lower(): i for i, lab in enumerate(labels)}
    found_indices = []
    
    for channel_name in test_channels:
        idx = lower_map.get(channel_name.lower())
        if idx is not None:
            print(f"  {channel_name:5s} -> Index {idx:2d} (Label: {labels[idx]})")
            found_indices.append(idx)
        else:
            print(f"  {channel_name:5s} -> NOT FOUND")
    
    print(f"\nIndices for {test_channels}: {found_indices}")


def test_fp1_fp2():
    """Specifically test where Fp1 and Fp2 channels are located."""
    from pylsl import StreamInlet, resolve_byprop
    
    print("\n" + "="*60)
    print("Testing Fp1 and Fp2 Channel Locations")
    print("="*60)
    
    print("Looking for EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=5.0)
    if not streams:
        print("No EEG stream found!")
        return
    
    stream_info = streams[0]
    labels = get_channel_labels(stream_info, debug=True)
    
    print(f"Stream: {stream_info.name()}")
    print(f"Total channels: {stream_info.channel_count()}\n")
    
    lower_map = {lab.lower(): i for i, lab in enumerate(labels)}
    
    # Check Fp1
    fp1_idx = lower_map.get('fp1')
    if fp1_idx is not None:
        print(f"✓ Fp1 found at Index {fp1_idx:2d} (Label: {labels[fp1_idx]})")
    else:
        print(f"✗ Fp1 NOT FOUND")
        print(f"  Available labels: {labels}")
    
    # Check Fp2
    fp2_idx = lower_map.get('fp2')
    if fp2_idx is not None:
        print(f"✓ Fp2 found at Index {fp2_idx:2d} (Label: {labels[fp2_idx]})")
    else:
        print(f"✗ Fp2 NOT FOUND")
        print(f"  Available labels: {labels}")
    
    if fp1_idx is not None and fp2_idx is not None:
        print(f"\nBlink detection picks (--picks Fp1,Fp2) will use indices: {fp1_idx},{fp2_idx}")


if __name__ == '__main__':
    test_channel_picks()
    test_fp1_fp2()


def parse_picks(picks_arg):
    if not picks_arg:
        return None, None
    parts = [p.strip() for p in picks_arg.split(',') if p.strip()]
    is_index = all(p.isdigit() for p in parts)
    if is_index:
        return [int(p) for p in parts], None
    return None, parts


def resolve_picks(info, picks_arg, default_names=('Cz', 'C3', 'C4')):
    labels = get_channel_labels(info)
    idxs, names = parse_picks(picks_arg) if picks_arg else (None, None)
    if idxs is not None:
        return idxs
    lower_map = {lab.lower(): i for i, lab in enumerate(labels)}
    chosen = []
    names_to_use = names if names else list(default_names)
    for nm in names_to_use:
        i = lower_map.get(nm.lower())
        if i is not None:
            chosen.append(i)
    if len(chosen) == len(names_to_use) and chosen:
        return chosen
    # fallback first 3
    return list(range(min(3, info.channel_count())))
