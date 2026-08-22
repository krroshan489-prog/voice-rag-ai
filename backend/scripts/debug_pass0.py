import re, sys
sys.path.insert(0, '.')
from backend.app.guardrails.verifier import GuardrailVerifier

query = 'benefits of earth certificate'
STOP = {'what','when','where','which','that','this','with','from','have','been','will','were','they',
        'their','then','than','into','your','also','some','more','does','give','make','tell','show',
        'just','very','hour','hours','time','times','days','year','years','week','weeks','month',
        'minute','minutes','second','number','amount','total','many','much','most','would','could',
        'should','about','after','before','study','studies','learn','learning','need','want'}
words = [w for w in re.findall(r'\b\w{4,}\b', query.lower()) if w not in STOP]
stems = {w[:5] for w in words}
print('Query words:', words)
print('Query stems:', stems)

# Simulate SPHR context
context = ('The Benefits of Obtaining SPHR Certification. SPHR certification is a credential that '
           'validates your expertise in human resources management and strategic HR planning. '
           'Earth is a common word that sometimes appears. certificate certif')
ctx_words = re.findall(r'\b\w{4,}\b', context.lower())
ctx_stems = {w[:5] for w in ctx_words if w not in STOP}
matched = stems & ctx_stems
print('Context stems (sample):', sorted(ctx_stems)[:20])
print('Matched:', matched)
print('Ratio:', round(len(matched)/len(stems), 3) if stems else 0)
print('Abs matches:', len(matched))
print('MIN_ABSOLUTE_MATCHES:', GuardrailVerifier.MIN_ABSOLUTE_MATCHES)
print('RELEVANCE_THRESHOLD:', GuardrailVerifier.RELEVANCE_THRESHOLD)
ratio = len(matched)/len(stems) if stems else 0
abs_ok = len(matched) >= GuardrailVerifier.MIN_ABSOLUTE_MATCHES
ratio_ok = ratio >= GuardrailVerifier.RELEVANCE_THRESHOLD
print('Would PASS ratio gate?', ratio_ok)
print('Would PASS abs gate?', abs_ok)
print('Would PASS overall?', ratio_ok and abs_ok)
