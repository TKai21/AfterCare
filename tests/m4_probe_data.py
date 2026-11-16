"""Probe data from the M4 re-review (scratchpad/rereview2), kept as project tests."""
AFF = {
    "chest pain": "I have chest pain",
    "chest pressure": "my chest feels tight",
    "trouble breathing": "I have trouble breathing",
    "gasping": "I am gasping",
    "choking": "I am choking",
    "collapsed": "I collapsed",
    "trouble swallowing": "I have trouble swallowing",
    "sudden vision loss": "I have sudden vision loss",
    "shortness of breath": "I am short of breath",
    "short of breath": "I am short of breath",
    "can't breathe": "I can't breathe",
    "cannot breathe": "I cannot breathe",
    "face drooping": "my face is drooping",
    "stroke": "I think it is a stroke",
    "slurred speech": "I have slurred speech",
    "weakness on one side": "I have weakness on one side",
    "numbness on one side": "my arm is numb",
    "worst headache": "worst headache of my life",
    "fainted": "I fainted",
    "passed out": "I passed out",
    "unconscious": "he is unconscious",
    "coughing up blood": "I am coughing up blood",
    "vomiting blood": "I am vomiting blood",
    "heavy bleeding": "I have heavy bleeding",
    "bleeding that won't stop": "the bleeding won't stop",
    "seizure": "I had a seizure",
    "suicidal": "I feel suicidal",
    "want to die": "I want to die",
    "kill myself": "I want to kill myself",
    "self harm": "I want to self harm",
    "overdose": "I took an overdose",
    "severe allergic reaction": "I am having an allergic reaction",
    "anaphylaxis": "I think this is anaphylaxis",
    "swelling of the throat": "my throat is closing",
    "throat closing": "my throat is closing",
    "face swelling": "my lips are swollen",
    "blue lips": "I have blue lips",
    "heart attack": "I think I am having a heart attack",
    "fast or irregular heartbeat": "my heart is racing",
    "vomiting that won't stop": "the vomiting won't stop",
    "can't keep fluids down": "I can't keep fluids down",
    "can't urinate": "I can't urinate",
}

# The re-review's wrappers: each puts the affirmative phrase next to a denial
# or a hedge. A reply wrapped this way must keep every emergency term.
WRAPS = [
    "{p}", "No fever, but {p}.", "No fever but {p}", "no, {p}", "No. {p}.", "not sure, {p}",
    "No fever. {p}.", "I don't have a fever and {p}", "Nope {p}", "No I don't, {p}",
    "never mind the fever, {p}", "No, not at all, {p}", "No fever or chills, {p}",
    "Not really. {p}", "None of those. {p}", "No fever; {p}", "Not anymore, {p} now",
    "I don't know, {p}",
]

# Denials next to a synonym the label's own words do not cover (N1).
SYNONYMS = [
    "No heavy bleeding, but it is soaking through the bandage",
    "No heavy bleeding. I am hemorrhaging",
    "No heavy bleeding, but there is blood in my urine",
    "No seizure, but he is convulsing",
    "Not unconscious, but he is unresponsive",
    "He is not unconscious, he is not responding",
    "No face swelling, but my lips are swollen",
    "No face swelling. My tongue is swollen",
    "No blue lips, but his face is purple",
    "No numbness on one side, but my arm is numb",
    "No weakness on one side, I am weak in my left arm",
    "No sudden vision loss but I am going blind",
    "No heart attack, cardiac arrest",
    "No trouble breathing but he is not breathing",
]

# Retractions and repeats (N2).
RETRACTIONS = [
    "No chest pain. Actually yes I do", "No chest pain, wait, I do", "No chest pain? Yes",
    "No chest pain, I lied", "I did not faint. Well I did", "No fainting... ok I did",
    "No chest pain - chest pain", "no chest pain chest pain",
]

PLAIN_DENIALS = ["No chest pain.", "No fever and no chest pain.", "No chest pain at all",
                 "no seizures"]
