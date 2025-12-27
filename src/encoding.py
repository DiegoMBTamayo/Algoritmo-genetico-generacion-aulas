"""
Codifica y decodifica el cromosoma de 30 bits:
Docente(6) | Días(6) | Aula1(4) | Hora1(5) | Aula2(4) | Hora2(5)
"""
from dataclasses import dataclass
from typing import Tuple, List
from .model import Gene, CourseOffering
from .domains import split_weekly_hours


@dataclass(frozen=True)
class ChromosomeBits:
    teacher_bits: str
    day_bits: str
    room1_bits: str
    start1_bits: str
    room2_bits: str
    start2_bits: str

    def as_string(self) -> str:
        return (
            f"{self.teacher_bits}{self.day_bits}"
            f"{self.room1_bits}{self.start1_bits}"
            f"{self.room2_bits}{self.start2_bits}"
        )


def _to_bin(val: int, bits: int) -> str:
    return format(max(0, int(val)), f"0{bits}b")


def days_to_mask(days: Tuple[int, ...]) -> str:
    mask = ["0"] * 6
    for d in days[:6]:
        if 0 <= d < 6:
            mask[d] = "1"
    return "".join(mask)


def mask_to_days(mask: str) -> Tuple[int, ...]:
    return tuple(i for i, ch in enumerate(mask[:6]) if ch == "1")


def gene_to_bits(g: Gene) -> ChromosomeBits:
    return ChromosomeBits(
        teacher_bits=_to_bin(g.teacher_id, 6),
        day_bits=days_to_mask(g.days),
        room1_bits=_to_bin(g.room1 if g.room1 is not None else 0, 4),
        start1_bits=_to_bin(g.start1 if g.start1 is not None else 0, 5),
        room2_bits=_to_bin(g.room2 if g.room2 is not None else 0, 4),
        start2_bits=_to_bin(g.start2 if g.start2 is not None else 0, 5),
    )


def bits_to_gene(bits: str, offer: CourseOffering, teacher_default: int = 0) -> Gene:
    """
    Decodifica un string de 30 bits en un Gene usando la regla de bloques (ROG6).
    """
    clean = bits.replace(" ", "")
    if len(clean) != 30:
        raise ValueError("El cromosoma debe tener 30 bits")
    teacher = int(clean[0:6], 2)
    day_mask = clean[6:12]
    room1 = int(clean[12:16], 2)
    start1 = int(clean[16:21], 2)
    room2 = int(clean[21:25], 2)
    start2 = int(clean[25:30], 2)

    days = mask_to_days(day_mask)
    h1, h2 = split_weekly_hours(offer.weekly_hours)

    if len(days) == 0:
        days = (0,) if h2 == 0 else (0, 1)
    elif len(days) == 1 and h2 > 0:
        days = (days[0], min(5, days[0] + 1))

    gene_kwargs = dict(
        teacher_id=teacher or teacher_default,
        days=days[:2],
        room1=room1,
        start1=start1,
        len1=h1,
        room2=None,
        start2=None,
        len2=None,
    )

    if h2 > 0:
        gene_kwargs.update({"room2": room2, "start2": start2, "len2": h2})

    return Gene(**gene_kwargs)
