from datetime import datetime
from zoneinfo import ZoneInfo

DIAS_SEMANA = {
    0: "segunda-feira", 1: "terça-feira", 2: "quarta-feira",
    3: "quinta-feira", 4: "sexta-feira", 5: "sábado", 6: "domingo",
}
MESES = {
    1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio",
    6: "junho", 7: "julho", 8: "agosto", 9: "setembro", 10: "outubro",
    11: "novembro", 12: "dezembro",
}


def build_base_system_prompt(pinned_memories: list[dict] = None) -> str:
    now = datetime.now(ZoneInfo("America/Sao_Paulo"))
    dia_semana = DIAS_SEMANA[now.weekday()]
    mes = MESES[now.month]
    data_por_extenso = f"{dia_semana}, {now.day:02d} de {mes} de {now.year}, {now.strftime('%H:%M')}"
    prompt = (
        f"Data e hora atual: {data_por_extenso}.\n"
        f"Localização padrão do usuário (salvo indicação contrária): Lages, "
        f"Santa Catarina, Brasil.\n"
        f"Responda sempre em português do Brasil, a menos que o usuário peça "
        f"explicitamente outro idioma."
    )
    if pinned_memories:
        memoria_section = "\n\n## Memória persistente (sempre disponível)\n"
        for entry in pinned_memories:
            tags_str = ", ".join(entry["tags"]) if entry["tags"] else "sem tags"
            memoria_section += f"- [{tags_str}] {entry['content']}\n"
        prompt += memoria_section
    return prompt
