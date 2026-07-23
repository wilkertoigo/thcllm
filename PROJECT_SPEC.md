# PROJECT_SPEC.md — THC CLI / THC LLM
Documento único de verdade do projeto. O Kilo Code deve ler este arquivo
INTEIRO antes de iniciar qualquer tarefa, e revisitar a seção de Guardrails
e Lições Aprendidas durante a execução. Se algo aqui divergir do código
real, pare e reporte — não assuma, não invente.

Última atualização: 2026-07-23 — pós Fase 5 (fallback automático de providers)

---

## 1. Stack técnica

- Linguagem: Python 3 (Ubuntu Server 24 LTS + LXQt)
- Libs principais da CLI: rich (UI/terminal), tiktoken (contagem real de tokens),
  pexpect (testes de terminal real), unittest (framework de testes padrão)
- 5 providers de LLM: thc (servidor próprio HF Space), groq, mistral, gemini, openrouter
- Servidor THC LLM: Hugging Face Space, produção em
  https://hulktoigo-thcllm.hf.space (NUNCA usar huggingface.co/spaces/..., quebra OAuth)
- Auth do servidor: header X-THC-Key
- Execução da CLI: sempre `cd ~/thcllm && python3 -m thc_cli <comando>`
- Config do usuário: ~/.thcrc (lido via config.get("chave", default))

## 2. Arquitetura (não inventar caminho — confirmar sempre com ls/cat antes)

~/thcllm/thc_cli/
main.py, main.py
core/
client.py, config.py, ui.py, agent.py, session.py, system_prompt.py, plan.py, tokens.py
tools/ (file_tools.py, bash_tool.py, web_tools.py, todo_tool.py)
providers/ (thc_provider.py, groq_provider.py, mistral_provider.py,
gemini_provider.py, openrouter_provider.py)
commands/
chat.py, models.py, quota.py, code.py
tests/
test_robustez_fase1b_2_1.py, test_agente_real_pexpect.py, test_tokens.py


- `DESTRUCTIVE_TOOLS = {"write_file", "str_replace", "bash"}` — sempre exigem
  confirmação do usuário antes de executar (via `confirm()`), sem exceção.
- `TOOLS_BY_NAME` é o registro automático de tools — nunca hardcodar nome de
  tool em outro lugar sem checar esse dict primeiro.
- `run_agent()` em agent.py é o loop central do modo agente. Qualquer novo
  callback (como `on_round_complete`) deve ter default `None` e não alterar
  comportamento existente quando não usado — testado explicitamente.

## 3. Guardrails obrigatórios para o Kilo

1. Antes de codar: ler os arquivos reais envolvidos (cat/grep/view). Nunca
   assumir assinatura de função, nome de config, ou comportamento existente
   sem confirmar no código.
2. Toda tool destrutiva (write_file, str_replace, bash) passa por preview()
   + confirmação do usuário. Isso não é negociável e não pode ser
   contornado "só para testar".
3. Toda mudança em agent.py precisa vir com teste que prove que o
   comportamento ANTIGO (sem o parâmetro/callback novo) continua idêntico.
4. Nunca usar `return` dentro do loop do agente para erros recuperáveis —
   usar `continue` (ver Fase 1-B, parser resiliente).
5. Nenhuma tarefa é considerada concluída sem: saída completa do
   `python3 -m unittest discover tests -v`, diff real (`git diff`) dos
   arquivos tocados, e conteúdo completo de qualquer arquivo novo.
6. Streaming (Fase 2.2) está deliberadamente adiado — não implementar nada
   de streaming "de brinde" em outras fases.
7. Respostas e system prompts da CLI são sempre em português
   (ver core/system_prompt.py, fuso America/Sao_Paulo).

## 4. Roadmap vivo

- [x] Fase 0 — Fundação
- [x] Fase 1-A — Provider abstraction
- [x] Fase 1-B — Robustez do core (parser resiliente, sessões, tokens reais)
- [x] Fase 2.1 — Diff/preview antes de aplicar edits + status persistente
- [x] Fase 2.3 — Plan Mode (/plan, generate_plan(), on_round_complete,
      plan_mode_step_confirm em ~/.thcrc) — 50 testes OK
- [ ] Fase 2.2 — Streaming (rich.live.Live) — ADIADA DE PROPÓSITO para o final
- [x] Fase 3 — Memória de longo prazo (MemoryStore, memory_write/read tools,
      pinned entries, injetadas automaticamente no system prompt)
      commit: f7583f1
- [x] Fase 4 — Skills plug-and-play (SkillStore, /skills REPL, --skill CLI,
      tools_allowed, system_prompt_extra) — 18 testes
      commit: 8f1edf5
- [x] Fase 5 — Provider padrão + fallback automático por rate limit
      (default openrouter, ProviderRateLimitError, chat_completion_with_fallback,
      ordem configurável em ~/.thcrc, thc fora da cadeia automática)
- [ ] Fase 6 — MCP (MCPTool, config em ~/.thcrc)
- [ ] Fase 7 — Sub-agentes/orquestração (AgentTool, TaskCreateTool)
- [ ] Fase 8 — Integrações de sistema (browser, LSP, git worktree)
- [ ] Fase 9 — Empacotamento (pip install thc-cli, testes, CI)

## 5. Lições aprendidas (atualizado pelo Claude a cada rodada)

- Fase 1: 413 do Groq não era bug de payload, era rate limit de TPM do tier
  free — sempre checar limites reais do provider antes de assumir bug de código.
- Fase 2.3: nenhum deslize registrado nesta rodada. Diff, testes e módulo
  novo (plan.py) vieram exatamente como especificado.
- Fase 5: diagnóstico do teste `test_alucinacao_fallback_llama_8b` foi só
  parcialmente confirmado — as 3 execuções de verificação reproduziram apenas
  rate limit (skip), não o cenário original de resposta malformada em bloco de
  código. O parser de fallback ainda não foi validado contra esse formato
  específico.

## 6. Ethos compartilhado (injetado no system prompt de todo provider)

Este texto entra como constante em core/system_prompt.py e é concatenado
ao system prompt de QUALQUER provider chamado pela CLI — garante o mesmo
tom e postura independente de qual dos 5 modelos está respondendo.
Não implica comunicação real entre modelos (cada chamada é isolada); é

STATUS ATUAL: apenas decorativo (docstring). NÃO concatenar ao system prompt
real ainda. Planejado para o futuro: integração via RAG, com flag de
desativar fácil em ~/.thcrc (ex: ethos_enabled = false). Não implementar
essa integração sem instrução explícita nova.
consistência de identidade imposta pelo framework, não telepatia entre LLMs.

