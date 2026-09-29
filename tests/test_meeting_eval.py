from tools.evaluate_meeting_ami import edit_distance, map_speakers, normalize_words, parse_timed_text
from src.stt.diarizer import SpeakerTurn


def test_parse_ami_timed_speaker_reference():
    value = "<|0.00|> Hello there.<|1.20|><|2.50|> Yes.<|2.90|>"
    assert parse_timed_text(value) == [
        (0.0, 1.2, "Hello there."),
        (2.5, 2.9, "Yes."),
    ]


def test_meeting_eval_maps_local_diarizer_ids_by_timed_overlap():
    turns = [
        SpeakerTurn(("speaker_1",), 0.0, 1.0, None),
        SpeakerTurn(("speaker_0",), 2.0, 3.0, None),
    ]
    references = {
        "speaker1": [(2.0, 3.0, "I agree")],
        "speaker2": [(0.0, 1.0, "That works")],
    }
    assert map_speakers(turns, references) == {
        "speaker_0": "speaker1",
        "speaker_1": "speaker2",
    }


def test_meeting_eval_maps_four_speakers_in_natural_conversation():
    turns = [
        SpeakerTurn((f"local_{index}",), float(index * 2), float(index * 2 + 1), None)
        for index in range(4)
    ]
    references = {
        f"person_{index}": [(float((3 - index) * 2), float((3 - index) * 2 + 1), "words")]
        for index in range(4)
    }

    assert map_speakers(turns, references) == {
        "local_0": "person_3",
        "local_1": "person_2",
        "local_2": "person_1",
        "local_3": "person_0",
    }


def test_meeting_eval_uses_normalized_word_error_rate():
    reference = normalize_words("Well, fifty percent of them.")
    hypothesis = normalize_words("well 50 percent of them")
    assert edit_distance(reference, hypothesis) == 1
