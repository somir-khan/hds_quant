def get_bits_usage(architecture, config, group_size=128):
    # Upstream effective-bit convention; adapter reports payload separately.
    return sum(config['linear_shape'][name][0] * config['linear_shape'][name][1]
               * (bit + 32 / group_size) for name, bits in architecture['linear'].items()
               for bit in bits) / config['model_numel']
