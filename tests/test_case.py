from gpt_5_2_reasoning_ablation.case import ClinicalCase


def test_reasoning_checklist_splits_and_trims():
    case = ClinicalCase(
        pmcid="PMC1",
        case_prompt="Prompt",
        diagnostic_reasoning="First clue. Second clue! Third clue? Final note; extra detail.",
        final_diagnosis="Diagnosis",
    )

    checklist = case.reasoning_checklist(max_items=3)
    assert checklist == ["First clue.", "Second clue!", "Third clue?"]


def test_reasoning_checklist_normalizes_numbered_rubric_fragments():
    case = ClinicalCase(
        pmcid="PMC2",
        case_prompt="Prompt",
        diagnostic_reasoning=(
            "1. Progressive unilateral ptosis strongly supports oculomotor nerve involvement. "
            "2. Pupil-sparing pattern argues against compressive aneurysm. "
            "3. Normal inflammatory markers make giant-cell arteritis unlikely. "
            "4. No fluctuating fatigability, so myasthenia gravis is less likely."
        ),
        final_diagnosis="Diagnosis",
    )

    checklist = case.reasoning_checklist()
    assert 3 <= len(checklist) <= 6
    assert all(not item.startswith(tuple(str(i) for i in range(10))) for item in checklist)
    assert all(len(item) <= 220 for item in checklist)


def test_reasoning_checklist_normalizes_unicode_punctuation():
    case = ClinicalCase(
        pmcid="PMC3",
        case_prompt="Prompt",
        diagnostic_reasoning='Pupil-sparing pattern \u2014 argues against aneurysm. \u201cNormal ESR\u201d lowers arteritis suspicion.',
        final_diagnosis="Diagnosis",
    )
    checklist = case.reasoning_checklist()
    assert all("\\u201" not in item for item in checklist)
    assert all("\u2014" not in item for item in checklist)
    assert all("\u201c" not in item and "\u201d" not in item for item in checklist)


def test_reasoning_checklist_strips_trailing_quoted_fragments():
    case = ClinicalCase(
        pmcid="PMC4",
        case_prompt="Prompt",
        diagnostic_reasoning=(
            'Mass lesion excluded by MRI - "MRI showed no mass lesion. '
            "Hypokalemia with resistant hypertension supports mineralocorticoid excess."
        ),
        final_diagnosis="Diagnosis",
    )
    checklist = case.reasoning_checklist()
    assert any("Mass lesion excluded by MRI" == item for item in checklist)
    assert all(' - "' not in item for item in checklist)


def test_reasoning_checklist_splits_escaped_numbered_fragments_and_drops_debris():
    case = ClinicalCase(
        pmcid="PMC5",
        case_prompt="Prompt",
        diagnostic_reasoning=(
            '1. Serous retinal detachment raised concern for nephrotic syndrome and SRC was suspected.\\" '
            "2. VEGF-blockade-induced thrombotic microangiopathy remained in the differential because intravitreal "
            "bevacizumab can cause endothelial injury and TMA. "
            'Acute cholecystitis was suspected but lacked gallstones: "no stone, and no nodule were the '
            'Gallbladder volvulus suggested by cone-shaped neck finding and CT findings of'
        ),
        final_diagnosis="Diagnosis",
    )
    checklist = case.reasoning_checklist()
    assert any("VEGF-blockade-induced thrombotic microangiopathy" in item for item in checklist)
    assert all('\\"' not in item for item in checklist)
    assert all(not item.endswith("CT findings of") for item in checklist)
    assert all(not item.endswith("were the") for item in checklist)
