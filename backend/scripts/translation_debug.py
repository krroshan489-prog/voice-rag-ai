"""Diagnose Tamil translation output."""
import sys, os, json, urllib.request
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.app.utils.translation import translate_query_to_english

tamil = '\u0bae\u0bc7\u0baa\u0bc7\u0baa\u0bcd \u0b9a\u0bbe\u0ba9\u0bcd\u0bb1\u0bbf\u0ba4\u0bb4\u0bcd \u0b8e\u0ba9\u0bcd\u0bb1\u0bbe\u0bb2\u0bcd \u0b8e\u0ba9\u0bcd\u0ba9?'
result = translate_query_to_english(tamil)

OUT = open(os.path.join(os.path.dirname(__file__), 'translation_debug.txt'), 'w', encoding='utf-8')
OUT.write('Tamil input  : ' + tamil + '\n')
OUT.write('Translated to: ' + result + '\n\n')

# Also hit the live API and capture translated_query from debug
body = json.dumps({'query': tamil, 'chunking_strategy': 'recursive',
                   'top_k': 4, 'stt_latency_ms': 0}).encode()
req = urllib.request.Request('http://localhost:8000/api/query', data=body,
                              headers={'Content-Type': 'application/json'}, method='POST')
resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
OUT.write('API translated_query : ' + str(resp['debug'].get('translated_query')) + '\n')
OUT.write('API was_translated   : ' + str(resp['debug'].get('was_translated')) + '\n')
OUT.write('guardrail_status     : ' + str(resp['debug']['guardrail_status']) + '\n')
OUT.write('can_answer           : ' + str(resp['can_answer']) + '\n')
OUT.write('confidence           : ' + str(resp.get('confidence')) + '\n')
OUT.write('answer               : ' + resp.get('answer','')[:200] + '\n')
OUT.write('sources              : ' + str(resp.get('sources', [])[:3]) + '\n')
OUT.write('msmarco_sources qids : ' + str([s.get('query_id') for s in resp.get('msmarco_sources', [])[:3]]) + '\n')
OUT.close()
