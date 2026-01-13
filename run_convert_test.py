import helpers, os
src=r"C:/fastapi_server/data/samples/xin_chao.mp3"
dst=r"C:/fastapi_server/data/tmp/test_conv_xin_chao.wav"
os.makedirs(os.path.dirname(dst), exist_ok=True)
try:
    helpers.convert_to_wav16_mono(src,dst)
    print('CONVERT_OK', dst)
except Exception as e:
    print('CONVERT_ERROR', str(e))
