import unittest
from src.encoding import gene_to_bits, bits_to_gene
from src.model import Gene, Individual, CourseOffering
from src.domains import OfferDomain, split_weekly_hours
from src.operators import repair_individual, interleaved_crossover, mutate_swap
from src.evaluation import evaluate
from src.config import GAConfig


class EncodingTests(unittest.TestCase):
    def test_gene_roundtrip(self):
        off = CourseOffering(
            cod_curso="C01",
            nombre="Test",
            ciclo=1,
            turno="M",
            grupo_horario="01S",
            tipo_hora="T",
            weekly_hours=4,
            is_lab=False,
        )
        gene = Gene(
            teacher_id=3,
            days=(0, 2),
            room1=1,
            start1=2,
            len1=2,
            room2=2,
            start2=5,
            len2=2,
        )
        bits = gene_to_bits(gene)
        decoded = bits_to_gene(bits.as_string(), off)
        self.assertEqual(decoded.teacher_id, 3)
        self.assertEqual(decoded.days[0], 0)
        self.assertEqual(decoded.start1, 2)
        self.assertEqual(decoded.room2, 2)


class EvaluationTests(unittest.TestCase):
    def test_penalty_conflicts(self):
        cfg = GAConfig()
        off1 = CourseOffering("C01", "X", 1, "M", "01S", "T", 2, False)
        off2 = CourseOffering("C02", "Y", 1, "M", "01S", "T", 2, False)
        g1 = Gene(teacher_id=1, days=(0,), room1=0, start1=0, len1=2)
        g2 = Gene(teacher_id=1, days=(0,), room1=0, start1=0, len1=2)
        ind = Individual([g1, g2])
        res = evaluate(ind, [off1, off2], n_cycles=2, n_rooms=2, n_teachers=2, cfg=cfg)
        self.assertEqual(res.hi, 2)
        self.assertEqual(res.ai, 2)
        self.assertEqual(res.pi, 2)
        self.assertEqual(ind.penalty, 22)


class OperatorTests(unittest.TestCase):
    def test_repair_respects_turn_slots(self):
        cfg = GAConfig()
        off = CourseOffering("C01", "X", 1, "T", "01S", "T", 2, False)
        dom = OfferDomain(
            teacher_ids=[1],
            room_ids=[0],
            allowed_day_idxs=[0, 1],
            allowed_start_slots=[6],
            fixed_assignment=None,
            turn="T",
        )
        bad_gene = Gene(teacher_id=1, days=(0,), room1=0, start1=0, len1=2)
        ind = Individual([bad_gene])
        repaired = repair_individual(ind, [off], {0: dom}, cfg)
        self.assertIn(repaired.genes[0].start1, dom.allowed_start_slots)

    def test_interleaved_crossover_and_mutation(self):
        cfg = GAConfig()
        off = CourseOffering("C01", "X", 1, "M", "01S", "T", 4, False)
        g1 = Gene(teacher_id=1, days=(0, 1), room1=0, start1=1, len1=2, room2=1, start2=3, len2=2)
        g2 = Gene(teacher_id=2, days=(2, 3), room1=2, start1=4, len1=2, room2=3, start2=6, len2=2)
        c1 = Individual([g1])
        c2 = Individual([g2])
        child = interleaved_crossover(c1, c2)
        self.assertNotEqual(child.genes[0].days, g1.days)
        mutate_swap(child, mutation_rate=1.0)
        self.assertEqual(child.genes[0].days[0], g2.days[1])


if __name__ == "__main__":
    unittest.main()
