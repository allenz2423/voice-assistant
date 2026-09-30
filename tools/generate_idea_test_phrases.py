"""Generate the deterministic 2,000-utterance idea-router evaluation corpus."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "data" / "idea_router_phrases_2000.jsonl"


def unique_rows():
    rows: dict[str, str] = {}
    route_counts = {"direct_action": 0, "personal_calendar_commitment": 0, "none": 0}

    def add(route: str, phrases: list[str], limit: int) -> None:
        for phrase in phrases:
            normalized = " ".join(phrase.split())
            if normalized and normalized not in rows and route_counts[route] < limit:
                rows[normalized] = route
                route_counts[route] += 1

    apps = [
        "Spotify", "Discord", "Firefox", "Chrome", "Edge", "Steam", "OBS", "the terminal",
        "my browser", "the file manager", "VS Code", "Zed", "Telegram", "Signal", "Slack",
        "YouTube Music", "VLC", "the calculator", "Settings", "the music player", "the downloads folder",
        "Cyberpunk", "Counter-Strike", "Hollow Knight", "DaVinci Resolve", "the calendar", "the notes app",
        "the workspace", "my email", "the camera", "the microphone", "the volume mixer",
        "the current window", "the previous tab", "the next workspace", "the weather app",
        "the system monitor", "the clipboard", "the screenshot tool", "the app launcher",
    ]
    command_templates = [
        "Open {app}.", "Launch {app}.", "Start {app}.", "Could you open {app}?",
        "Please launch {app}.", "Adam, open {app}.", "Switch to {app}.",
        "Bring up {app} for me.", "Can you start {app}?", "I want you to open {app}.",
        "Go to {app}.", "Run {app}.",
    ]
    command_phrases = [template.format(app=app) for app in apps for template in command_templates]
    add("direct_action", command_phrases, 250)

    action_phrases = [
        "Focus my browser", "Focus the Spotify window", "Switch my browser to the front",
        "Play my music", "Pause the video", "Skip this song", "Turn the volume down",
        "Turn up the volume", "Set a timer for ten minutes", "Remind me to call Alex at 4",
        "What time is it?", "What's the weather tomorrow?", "Show my upcoming calendar events",
        "Take a screenshot", "Show the open windows", "Find my latest download",
        "Search the web for the release date", "Put this window on workspace two",
        "Mute my microphone", "Copy that text", "Open a new browser tab", "Close this window",
        "Tell me the current CPU temperature", "Check whether the service is running",
        "Create a note called shopping", "Add milk to my shopping note", "Play the next episode",
        "Set a reminder for tomorrow morning", "What's playing right now?", "Show my battery level",
    ]
    command_phrases = [
        phrase if phrase.endswith(("?", ".")) else phrase + suffix
        for phrase in action_phrases
        for suffix in (".", " please.", " for me.", " right now.", " when you can.", "")
    ]
    direct_variants = []
    for phrase in command_phrases:
        direct_variants.extend((
            phrase,
            f"Hey Adam, {phrase[0].lower() + phrase[1:]}",
            f"Please, {phrase[0].lower() + phrase[1:]}",
            f"When you can, {phrase[0].lower() + phrase[1:]}",
        ))
    add("direct_action", command_phrases + direct_variants, 500)

    commitments = [
        "a meeting", "a doctor appointment", "a dentist appointment", "a call with my manager",
        "lunch with Jordan", "a job interview", "a class", "a therapy session", "a flight",
        "a project review", "a parent-teacher conference", "a work presentation", "a vet appointment",
        "a phone interview", "a team sync", "a concert", "a dinner reservation", "a study session",
        "a pickup appointment", "a consultation", "a deadline", "a checkup", "a planning session",
        "a performance", "a game", "an exam", "a family visit", "a call with my doctor",
        "a meeting with the design team", "a conference", "a video call",
    ]
    times = [
        "tomorrow at 3:30 pm", "tomorrow morning", "tomorrow afternoon", "this Friday at noon",
        "Friday evening", "next Monday at 9 am", "next week on Tuesday", "on October 14 at 2 pm",
        "in two days at 11", "later this afternoon", "the day after tomorrow at 4:15",
        "next month on the 5th", "this Wednesday morning", "Saturday at 6 pm", "on November 3",
        "in three weeks", "next Thursday at 1:30 pm", "at 10 am tomorrow", "on the 20th at noon",
        "Monday after lunch",
    ]
    commitment_templates = [
        "I have {event} {time}.", "I've got {event} {time}.",
        "My {event} is {time}.", "I'm scheduled for {event} {time}.",
        "I am going to {event} {time}.", "There's {event} on my schedule {time}.",
        "I will be at {event} {time}.", "I'm booked for {event} {time}.",
        "Put this in my plans: {event} {time}.", "I need to remember that I have {event} {time}.",
    ]
    calendar_phrases = [
        template.format(event=event, time=when)
        for event in commitments for when in times for template in commitment_templates
    ]
    add("personal_calendar_commitment", calendar_phrases, 500)

    negatives: list[str] = []
    past_templates = [
        "I had {event} {time}.", "My {event} was {time}.", "Yesterday I went to {event}.",
        "I finished {event} earlier today.", "Last week I had {event}.",
    ]
    past_times = ["yesterday", "last Friday", "this morning", "two days ago", "last month", "earlier today"]
    negatives.extend(t.format(event=e, time=tme) for e in commitments for tme in past_times for t in past_templates)

    hypothetical_templates = [
        "Maybe I should schedule {event} {time}.", "I might have {event} {time}.",
        "If I have {event} {time}, I'll let you know.", "I wonder whether I should book {event} {time}.",
        "Could be nice to plan {event} {time}.", "I don't know if I'll have {event} {time}.",
    ]
    negatives.extend(t.format(event=e, time=tme) for e in commitments for tme in times for t in hypothetical_templates)

    other_person_templates = [
        "My sister has {event} {time}.", "Jordan has {event} {time}.",
        "My coworker is going to {event} {time}.", "They scheduled {event} {time}.",
        "My partner is booked for {event} {time}.", "The video says someone has {event} {time}.",
    ]
    negatives.extend(t.format(event=e, time=tme) for e in commitments for tme in times for t in other_person_templates)

    discussion_phrases = [
        "I hate meetings.", "The meeting was really long.", "Do you know what time the meeting starts?",
        "The video is about a doctor appointment.", "I like Spotify, but I don't want to open it.",
        "My friend told me to open Discord.", "I was thinking about the calendar app.",
        "Maybe we should talk about meetings later.", "Spotify is playing a song I like.",
        "The browser is already open.", "I opened Steam yesterday.", "The dentist office called me earlier.",
        "I don't have any meetings today.", "I heard them say the appointment is tomorrow.",
        "What does the word appointment mean?", "Someone in this video is talking about a meeting.",
        "I'm not asking you to do anything, just thinking out loud.", "That reminds me of a calendar event from last year.",
        "I should probably get around to that sometime.", "It's nice outside today.",
        "The game update is taking a while.", "I wonder what time it is in Tokyo.",
        "The music is too loud in this video.", "My phone calendar has too many notifications.",
        "My brother's interview is next week.", "I might watch the meeting recording later.",
        "The calendar says the office is closed.", "I don't know if I have anything tomorrow.",
        "Open Spotify is the name of that playlist.", "They asked me whether I have an appointment.",
    ]
    discussion_variants = []
    for phrase in discussion_phrases:
        discussion_variants.extend((
            phrase,
            f"Earlier today, {phrase}",
            f"The person in the video said, {phrase}",
            f"I was just thinking this: {phrase}",
            f"Someone asked me this: {phrase}",
            f"I heard this sentence: {phrase}",
            f"To be clear, {phrase}",
            f"For example, {phrase}",
        ))

    # Keep five distinct negative families so the corpus checks intent, tense,
    # speaker ownership, quoted/media speech, and unrelated conversation.
    subjects = ["The weather", "That movie", "My laptop", "The new update", "The song", "The browser", "My neighbor", "The meeting"]
    predicates = ["looks pretty good", "was interesting", "sounds strange", "is taking forever", "might be useful", "was on sale", "feels too loud", "is hard to follow"]
    neutral = []
    for subject in subjects:
        for predicate in predicates:
            neutral.append(f"{subject} {predicate}.")
            for ending in ("today", "again", "in the video"):
                neutral.append(f"{subject} {predicate} {ending}.")
    def first_unique(phrases: list[str], count: int) -> list[str]:
        return list(dict.fromkeys(" ".join(phrase.split()) for phrase in phrases))[:count]

    balanced_negatives = (
        first_unique(negatives[:900], 200)
        + first_unique(negatives[900:4500], 200)
        + first_unique(negatives[4500:], 200)
        + first_unique(discussion_variants, 200)
        + first_unique(neutral, 200)
    )
    add("none", balanced_negatives, 1000)

    # Use a stable class-interleaved order to make corpus inspection easier.
    by_route = {route: [] for route in ("direct_action", "personal_calendar_commitment", "none")}
    for phrase, route in rows.items():
        by_route[route].append(phrase)
    selected = []
    for route, count in (("direct_action", 500), ("personal_calendar_commitment", 500), ("none", 1000)):
        selected.extend((phrase, route) for phrase in by_route[route][:count])
    if len(selected) != 2000:
        raise RuntimeError(f"Expected 2,000 phrases, got {len(selected)}")
    return selected


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for index, (phrase, route) in enumerate(unique_rows(), 1):
            handle.write(json.dumps({"id": index, "text": phrase, "expected": route}, ensure_ascii=False) + "\n")
    print(f"Wrote 2,000 evaluation phrases to {OUTPUT}")


if __name__ == "__main__":
    main()
