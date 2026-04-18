import requests
url = 'https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=584590&andmetyyp=lidar_laz_madal&dl=1&f=584590_2024_madal.laz&no_cache=69b4960b43dda&page_id=614'
try:
    r = requests.get(url, stream=True, timeout=30)
    print('status', r.status_code)
    print('headers:', r.headers.get('content-type'))
    print('content-length:', r.headers.get('content-length'))
    print('final-url:', r.url)
    r.close()
except Exception as e:
    print('error', e)
