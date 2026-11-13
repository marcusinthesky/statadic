"""The public evidence matrix includes aliases and keeps members distinct."""

import unittest

from bridge import ROOT, read, tracked_inputs
from evidence_crosswalk import crosswalk, validate_comparisons


class EvidenceCrosswalkTests(unittest.TestCase):
    def test_current_claim_symbols_belong_to_the_public_inventory(self):
        """Runtime-inherited methods cannot masquerade as inventoried members."""
        registry = ROOT / "src/lean/registry"
        public = read(registry / "public-api.json")
        names = {}
        for module in public["modules"]:
            for item in module["exports"]:
                name = f"{module['name']}.{item['name']}"
                names[name] = item["target"]
                for member in item.get("members", []):
                    names[f"{name}.{member['name']}"] = member["target"]
        targets = set(names.values())
        for claim in read(registry / "claims.json")["claims"]:
            for symbol in claim["python"]["symbols"]:
                with self.subTest(claim=claim["id"], symbol=symbol):
                    self.assertIn(names.get(symbol, symbol), targets)

    def test_aliases_and_members_only_inherit_identical_target_evidence(self):
        registry = ROOT / "src/lean/registry"
        public = read(registry / "public-api.json")
        comparisons = read(registry / "implementation-comparisons.json")["comparisons"]
        validate_comparisons(ROOT, comparisons)
        pinned_inputs = tracked_inputs()
        for case in comparisons.values():
            self.assertIn(case["pytest"].split("::", 1)[0], pinned_inputs)
        checked = {
            "id": "example",
            "proof_status": "kernel-checked-and-axiom-audited",
            "python": {"symbols": ["statadic.optimize.simplex.simplex_qp_result"]},
        }
        unchecked = {**checked, "id": "candidate", "proof_status": "unproved"}
        rows = crosswalk(public, [checked, unchecked], comparisons)
        by_name = {alias["name"]: row for row in rows for alias in row["public_names"]}
        expected = sum(
            1 + len(item.get("members", []))
            for module in public["modules"]
            for item in module["exports"]
        )
        self.assertEqual(len(by_name), expected)
        qp = by_name["statadic.optimize.simplex.simplex_qp_result"]
        self.assertIs(
            by_name["statadic.core.SolverResult"],
            by_name["statadic.core.results.SolverResult"],
        )
        self.assertEqual(qp["claims"], ["example"])
        self.assertEqual(qp["comparisons"], ["capped-simplex"])
        self.assertEqual(
            by_name["statadic.estimation.glm.PoissonMLE"]["comparisons"], []
        )
        self.assertEqual(
            by_name["statadic.estimation.glm.PoissonMLE.estimate"]["comparisons"],
            ["poisson-exposure"],
        )
        with self.assertRaisesRegex(ValueError, "symbol absent"):
            crosswalk(public, [], {"bad": {"symbols": ["statadic.misspelled"]}})
        with self.assertRaisesRegex(ValueError, "unknown pytest function"):
            case = dict(comparisons["capped-simplex"])
            case["pytest"] = case["pytest"].split("::")[0] + "::test_missing"
            validate_comparisons(ROOT, {"bad": case})
