import urllib.request, json, time

BASE = 'https://jarohullowicki-melusina-space.hf.space/gradio_api'

def call(msg):
    payload = json.dumps({'data': [msg, []]}).encode()
    req = urllib.request.Request(
        f'{BASE}/call/on_submit',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        resp = json.loads(r.read())
    event_id = resp['event_id']

    time.sleep(2)
    req2 = urllib.request.Request(f'{BASE}/call/on_submit/{event_id}')
    with urllib.request.urlopen(req2, timeout=30) as r:
        raw = r.read().decode()

    for line in reversed(raw.splitlines()):
        if line.startswith('data:'):
            try:
                return json.loads(line[5:].strip())
            except Exception:
                pass
    return None

TESTS = [
    ('Zwykla wiadomosc (NO_RULE)',   'Czym jest JFP?',                              'NO_RULE'),
    ('EXPLICIT zawsze',              'od teraz zawsze odpowiadaj po polsku',         'EXPLICIT'),
    ('EXPLICIT nigdy',               'nigdy nie uzywaj emoji w odpowiedziach',       'EXPLICIT'),
    ('EXPLICIT pamietaj',            'pamietaj ze nasz projekt nazywa sie VIKI',     'EXPLICIT'),
    ('DOMAIN to sie nazywa',         'to sie nazywa LoRA nie LORA',                  'DOMAIN'),
    ('DOMAIN nie mow...mowimy',      "nie mow fine-tuning mowimy dostrajanie",       'DOMAIN'),
    ('DOMAIN zamiast...uzywaj',      'zamiast LLM uzywaj model jezykowy',            'DOMAIN'),
    ('IMPLICIT korekta',             'wrong, nie uzywaj tabelek',                    'IMPLICIT'),
]

print('=' * 62)
print('  MELUSINA SPACE - testy funkcjonalne')
print('  https://jarohullowicki-melusina-space.hf.space')
print('=' * 62)
passed = failed = 0

for label, msg, expected in TESTS:
    try:
        data = call(msg)
        if data is None:
            raise ValueError('Brak odpowiedzi SSE')

        history     = data[0] if len(data) > 0 else []
        proposal_md = str(data[2]) if len(data) > 2 else ''
        audit_md    = str(data[7]) if len(data) > 7 else ''

        has_response = len(history) > 0

        if expected == 'NO_RULE':
            signal_ok = 'Brak wykrytego' in proposal_md or proposal_md.strip() == ''
        else:
            signal_ok = expected in proposal_md.upper()

        has_audit = 'JFP-' in audit_md
        ok = has_response and signal_ok and has_audit

        if ok:
            passed += 1
        else:
            failed += 1

        icon = 'PASS' if ok else 'FAIL'
        print(f'[{icon}] {label}')
        print(f'   sygnal  : {expected} -> {"OK" if signal_ok else "BLAD"}')
        print(f'   audit   : {"JFP-OK" if has_audit else "BRAK"}')
        print(f'   historia: {len(history)} msg')
        if not ok:
            print(f'   proposal: {proposal_md[:100]}')
            print(f'   audit_md: {audit_md[:100]}')
        print()
    except Exception as e:
        failed += 1
        print(f'[FAIL] {label}')
        print(f'   ERROR: {type(e).__name__}: {e}')
        print()

print('=' * 62)
print(f'  Wynik: {passed}/{passed+failed} PASS')
print('=' * 62)
