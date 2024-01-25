SDXL_BASE_CHANNEL_DICT = {
    'cond_output_channels': [320] * 4 + [640] * 3 + [1280] * 3,
    'project_channels': [160] * 4 + [320] * 3 + [640] * 3,
    'concat_channels': [320] * 2 + [640] * 3 + [1280] * 4 + [0]
}

SDXL_REFINE_CHANNEL_DICT = {
    'cond_output_channels': [384] * 4 + [768] * 3 + [1536] * 6,
    'project_channels': [192] * 4 + [384] * 3 + [768] * 6,
    'concat_channels': [384] * 2 + [768] * 3 + [1536] * 7 + [0]
}