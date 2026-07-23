# Manual Checklist — Robustez Fase 1-B + 2.1

Ambiente: `cd ~/thcllm`

## BLOCO 1 — Parser tool_call

### 1.1 Esgotamento de retry (mock)
```bash
python3 - <<'PY'
from unittest.mock import MagicMock
from thc_cli.core.agent import run_agent

provider = MagicMock()
provider.name = "thc"
provider.chat_completion.return_value = {
    "choices": [{"message": {"content": "texto qualquer sem tool_call válido"}}]
}

result = run_agent(
    provider=provider,
    messages=[{"role": "user", "content": "oi"}],
    max_rounds=2,
)
print("RESULTADO:", result)
```
**Esperado:** não crasha, retorna a última resposta do modelo sem tool_call.

### 1.2 Nome de tool inválido no fallback
```bash
python3 -c "
from thc_cli.core.agent import _extract_tool_calls
text = '\`\`\`read_file_xyz\n{\"name\": \"read_file_xyz\", \"arguments\": {\"path\": \"/tmp/x\"}}\n\`\`\`'
print(_extract_tool_calls(text))
"
```
**Esperado:** `[]` (não aceita tool inexistente).

### 1.3 JSON malformado no bloco alucinado
```bash
python3 -c "
from thc_cli.core.agent import _extract_tool_calls
text = '\`\`\`read_file\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/x\",}}\n\`\`\`'
print(_extract_tool_calls(text))
"
```
**Esperado:** `[]` (não lança exceção).

### 1.4 Dois blocos tool_call na mesma resposta
```bash
python3 -c "
from thc_cli.core.agent import _extract_tool_calls
text = '\`\`\`tool_call\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/a\"}}\n\`\`\`\`\`\`tool_call\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"/tmp/b\", \"content\": \"x\"}}\n\`\`\`'
print(_extract_tool_calls(text))
"
```
**Esperado:** 2 items. Documentar: no loop atual do `run_agent`, **ambos** são executados sequencialmente, não só o primeiro.

### 1.5 Modelo real com formato errado
```bash
echo "linha original" > /tmp/teste_fallback.txt
python3 -m thc_cli chat "edite o arquivo /tmp/teste_fallback.txt substituindo 'linha original' por 'linha modificada'" --agent --provider groq --model llama-3.1-8b-instant
```
**Esperado:** fallback reconhece bloco com nome de tool errado, mostra diff + painel de confirmação.

---

## BLOCO 2 — Sessões

### 2.1 /resume com JSON corrompido
```bash
mkdir -p ~/.thc/sessions
echo "json quebrado {" > ~/.thc/sessions/20260101-000000-0001.json
python3 -m thc_cli chat --provider thc --model ministral3b-mst
# dentro do REPL:
/resume 20260101-000000-0001
```
**Esperado:** erro tratado (mensagem clara), não traceback cru.

### 2.2 /sessions com pasta vazia
```bash
rm -f ~/.thc/sessions/*.json
python3 -m thc_cli chat --provider thc --model ministral3b-mst
# dentro do REPL:
/sessions
```
**Esperado:** mensagem amigável, sem traceback.

### 2.3 Salvar duas sessões com mesmo nome
```bash
python3 -m thc_cli chat --provider thc --model ministral3b-mst
# dentro do REPL:
/save
/save
```
**Esperado:** documentar comportamento atual (atualiza a mesma sessão).

### 2.4 /resume com ID inexistente
```bash
python3 -m thc_cli chat --provider thc --model ministral3b-mst
# dentro do REPL:
/resume 9999-inexistente
```
**Esperado:** "Sessão não encontrada: 9999-inexistente" em vermelho.

### 2.5 Sessão muito grande
```bash
python3 - <<'PY'
from thc_cli.core.session import save_session
msgs = [{"role": "user", "content": "msg " + str(i)} for i in range(250)]
save_session(None, msgs, "thc", "m", "medium")
PY
python3 -m thc_cli chat --provider thc --model ministral3b-mst
/resume <id-gerado>
```
**Esperado:** salva e recarrega sem perda de dados.

### 2.6 /resume mantém provider/model/mode
```bash
# Salvar sessão com --provider thc --model ministral3b-mst
# Depois /resume e observar banner
```
**Esperado:** banner reflete provider/model/mode da sessão.

---

## BLOCO 3 — Diff/preview

### 3.1 old_str duplicado (preview + confirmação)
```bash
echo "a b a" > /tmp/teste_dup.txt
python3 -m thc_cli chat "edite o arquivo /tmp/teste_dup.txt substituindo a primeira ocorrência de 'a' por 'z'" --agent --provider thc --model ministral3b-mst
```
**Esperado:** preview substitui só a primeira ocorrência; diff mostra `z b a`; confirmação aparece antes de escrever.

### 3.2 old_str não existe
```bash
echo "abc" > /tmp/teste_nao_existe.txt
python3 -m thc_cli chat "edite o arquivo /tmp/teste_nao_existe.txt substituindo 'x' por 'y'" --agent --provider thc --model ministral3b-mst
```
**Esperado:** erro claro "old_str não encontrado em /tmp/teste_nao_existe.txt", sem diff, sem escrita.

### 3.3 Arquivo binário
```bash
python3 -m thc_cli chat "edite o arquivo /bin/ls substituindo 'ELF' por 'XXX'" --agent --provider thc --model ministral3b-mst
```
**Esperado:** tratar graciosamente (não imprime diff binário ilegível).

### 3.4 Encoding não-UTF-8
```bash
printf '\xe9\x00' > /tmp/teste_latin1.bin
python3 -m thc_cli chat "edite o arquivo /tmp/teste_latin1.bin substituindo '\\xe9' por '\\xe9\\xe9'" --agent --provider thc --model ministral3b-mst
```
**Esperado:** tratar graciosamente ou documentar limitação.

### 3.5 Arquivo novo (write_file)
```bash
rm -f /tmp/teste_novo.txt
python3 -m thc_cli chat "crie o arquivo /tmp/teste_novo.txt com o conteúdo 'hello world'" --agent --provider thc --model ministral3b-mst
```
**Esperado:** diff mostra arquivo inteiro como adição, sem erro de leitura.

### 3.6 Cancelamento (usuário diz "não")
```bash
echo "linha original" > /tmp/teste_cancel.txt
python3 -m thc_cli chat "edite o arquivo /tmp/teste_cancel.txt substituindo 'linha original' por 'linha modificada'" --agent --provider thc --model ministral3b-mst
# quando aparecer "Permitir execução?", digite: n
```
**Esperado:** NADA escrito em disco; mensagem "Usuário negou permissão..." no resultado.

### 3.7 Diff de arquivo grande (2000+ linhas)
```bash
python3 - <<'PY'
from pathlib import Path
p = Path("/tmp/teste_grande.txt")
lines = ["linha " + str(i) + "\n" for i in range(2000)]
lines[1234] = "linha alterada\n"
p.write_text("".join(lines))
PY
python3 -m thc_cli chat "edite o arquivo /tmp/teste_grande.txt substituindo 'linha 999' por 'linha mil'" --agent --provider thc --model ministral3b-mst
```
**Esperado:** diff útil, não o arquivo inteiro; ou documentar se hoje imprime tudo.
