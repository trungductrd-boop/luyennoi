import audio_api, os, json
user_wav = r"C:/fastapi_server/data/tmp/test_conv_xin_chao.wav"
print('calling _handle_vocab_mode with', user_wav)
res = audio_api._handle_vocab_mode(user_wav, vocab_id='1_1', word='Xin chào', timeout=30, created_tmp=False)
print(json.dumps(res, ensure_ascii=False, indent=2))
