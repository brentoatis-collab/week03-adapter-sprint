#!/usr/bin/env python3
"""Deterministic generator for the CivicDesk 311 instruction dataset.

Produces complaint -> ticket pairs following the behavior specification in
README.md section 3. No external LLM is used; every example is assembled from
hand-written scenario content plus seeded perturbations.

Design (see README "Instruction dataset design"):
  * A FAMILY is one underlying issue scenario (e.g. "deep pothole"). Each
    category has 7 families. The train/eval split is made at the family level,
    so no held-out complaint shares its underlying issue with a training one.
  * Urgency is never hard-coded per phrase. Each issue and each optional context
    clause carries criteria FACTS (README 3.4); urgency = the highest-ranked fact.
    The same family therefore appears at different urgency levels depending on
    the stated context, and resident urgency claims ("URGENT!!", "no rush") are
    injected independently of the label so keyword lookup does not work.
  * Location text is protected from typo/slang noise (only casing changes), so
    the target never has to "fix" or guess an address.

Usage:
    python scripts/generate_instruction_data.py            # writes data/*.jsonl
    python scripts/generate_instruction_data.py --out-dir /tmp/x --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "adapter_config.json"

# ---------------------------------------------------------------------------
# Output contract (README 3.1)
# ---------------------------------------------------------------------------
TICKET_KEYS = ("category", "urgency", "location", "summary", "address_or_null")
CATEGORIES = [
    "pothole_road_damage", "streetlight", "garbage_illegal_dumping", "graffiti",
    "water_sewer", "tree_vegetation", "sidewalk_damage", "snow_ice", "noise",
    "abandoned_vehicle", "traffic_signage", "other",
]
URGENCY_LEVELS = ["low", "medium", "high", "emergency"]
CLARIFICATION_SENTENCE = "Location clarification required."
LOCATION_UNSPECIFIED = "unspecified"

# Urgency criteria facts (README 3.4) -> rank in URGENCY_LEVELS.
FACT_RANK = {
    "COSMETIC": 0,          # low: cosmetic / not time-sensitive
    "QUALITY_OF_LIFE": 1,   # medium: needs action within days, no immediate hazard
    "HAZARD_SOON": 2,       # high: likely injury/damage soon if not addressed
    "SERVICE_LOSS": 2,      # high: loss of an essential service
    "IMMEDIATE_DANGER": 3,  # emergency: immediate danger to life/safety
    "ACTIVE_DAMAGE": 3,     # emergency: property damage happening now
}

LOCATION_TYPES = ["exact_address", "intersection", "landmark", "block_range",
                  "street_only", "vague", "missing"]
NO_ADDRESS_TYPES = LOCATION_TYPES[1:]


def urgency_from_facts(facts: list[str]) -> str:
    return URGENCY_LEVELS[max(FACT_RANK[f] for f in facts)]


# ---------------------------------------------------------------------------
# Scenario content
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    issue: tuple[str, ...]   # full clauses, lowercase start ("there's a huge pothole")
    short: tuple[str, ...]   # terse noun phrases ("huge pothole")
    core: str                # neutral summary fragment
    fact: str                # baseline criteria fact
    point: bool = True       # point issue (needs a spot) vs street-wide issue


@dataclass(frozen=True)
class Context:
    key: str
    phrases: tuple[str, ...]
    summary: str | None      # neutral summary addition (None = not summarized)
    fact: str | None         # criteria fact added (None = no urgency effect)
    only: tuple[int, ...] | None = None  # scenario indices this context fits


S, C = Scenario, Context

SCENARIOS: dict[str, list[Scenario]] = {
    "pothole_road_damage": [
        S(("there's a huge pothole", "a massive hole opened up in the road", "there's a pothole that's like a foot deep"),
          ("huge pothole", "deep pothole"), "Deep pothole in the roadway", "QUALITY_OF_LIFE"),
        S(("there are a bunch of little potholes", "the lane is full of small potholes", "there's potholes all over the lane"),
          ("lots of small potholes", "potholes everywhere"), "Multiple small potholes in the roadway", "QUALITY_OF_LIFE"),
        S(("the pavement is all cracked and breaking up", "there are big cracks running across the road", "the asphalt is crumbling"),
          ("cracked pavement", "crumbling asphalt"), "Cracked and deteriorating pavement", "COSMETIC"),
        S(("a sinkhole opened up in the road", "part of the road has caved in", "a section of the travel lane collapsed"),
          ("sinkhole in road", "road caved in"), "Road surface collapse (sinkhole) in a travel lane", "IMMEDIATE_DANGER"),
        S(("the patch where they dug up the road has sunk", "the utility patch is sinking into a dip", "there's a big dip where the road was repaved"),
          ("sunken road patch", "dip in road patch"), "Sunken utility patch in the roadway", "QUALITY_OF_LIFE"),
        S(("the road is buckling", "a ridge of asphalt has pushed up across the lane", "the pavement heaved up into a bump"),
          ("road buckling", "pavement heave"), "Buckled/heaved pavement across the lane", "QUALITY_OF_LIFE"),
        S(("the edge of the road is crumbling away", "the shoulder is breaking off into the ditch", "the side of the road is washing out"),
          ("road edge washing out", "shoulder crumbling"), "Road edge crumbling/washing out", "QUALITY_OF_LIFE"),
    ],
    "streetlight": [
        S(("the streetlight is out", "the street light is dead", "the light on the pole isn't working"),
          ("streetlight out", "light out"), "Streetlight out", "QUALITY_OF_LIFE"),
        S(("the streetlight keeps flickering", "the light flickers on and off all night", "the street lamp is strobing"),
          ("flickering streetlight", "light flickering"), "Streetlight flickering", "QUALITY_OF_LIFE"),
        S(("the streetlight stays on all day", "the street light is on in broad daylight", "the lamp never turns off, even at noon"),
          ("streetlight on during day", "light on all day"), "Streetlight on during daytime", "COSMETIC"),
        S(("like four streetlights in a row are out", "the whole row of lights is dark", "several streetlights are out"),
          ("several lights out", "row of streetlights dark"), "Several consecutive streetlights out", "QUALITY_OF_LIFE"),
        S(("the light pole is leaning", "a streetlight pole is tilted way over", "the lamp post looks like it's about to tip"),
          ("leaning light pole", "tilted lamp post"), "Streetlight pole leaning", "HAZARD_SOON"),
        S(("a car knocked the light pole down", "the light pole got hit and is lying on the sidewalk", "the streetlight pole is down"),
          ("light pole knocked down", "pole down"), "Streetlight pole knocked down", "HAZARD_SOON"),
        S(("the access panel on the light pole is open with wires showing", "the cover at the base of the streetlight is missing and wires are exposed"),
          ("exposed wires at light pole", "open panel on pole"), "Exposed wiring at the base of a streetlight", "HAZARD_SOON"),
    ],
    "garbage_illegal_dumping": [
        S(("our trash didn't get picked up", "garbage pickup skipped our house", "the truck missed our bins again"),
          ("missed trash pickup", "trash not picked up"), "Missed garbage collection", "QUALITY_OF_LIFE"),
        S(("the whole street got skipped for recycling", "nobody on the street got recycling picked up", "recycling wasn't collected for the entire street"),
          ("recycling skipped whole street", "no recycling pickup on street"), "Recycling not collected for the whole street", "QUALITY_OF_LIFE", point=False),
        S(("someone dumped a mattress and a couch", "a pile of old furniture got dumped", "somebody left a bunch of junk furniture on the curb"),
          ("dumped furniture", "mattress and couch dumped"), "Furniture illegally dumped", "QUALITY_OF_LIFE"),
        S(("the public trash can is overflowing", "the bin at the bus stop is overflowing", "the city trash can is packed and spilling over"),
          ("overflowing public bin", "trash can overflowing"), "Public trash bin overflowing", "QUALITY_OF_LIFE"),
        S(("somebody dumped a pile of tires", "a stack of old tires got dumped", "there's like twenty tires dumped"),
          ("dumped tires", "pile of tires"), "Tires illegally dumped", "QUALITY_OF_LIFE"),
        S(("contractors dumped drywall and debris", "there's a pile of construction debris dumped", "someone dumped busted drywall and lumber"),
          ("construction debris dumped", "drywall dumped"), "Construction debris illegally dumped", "QUALITY_OF_LIFE"),
        # Localized: litter concentrated at one spot (street name alone cannot pinpoint it).
        S(("there's a big pile of litter heaped up in one spot", "wrappers and bottles are piled up in one corner by the curb",
           "a heap of trash and litter has built up next to one of the utility poles"),
          ("pile of litter in one spot", "litter heaped by curb"), "Excessive litter", "COSMETIC"),
    ],
    "graffiti": [
        S(("someone tagged the wall", "there's graffiti all over the wall", "somebody spray painted the side of the building"),
          ("graffiti on wall", "wall tagged"), "Graffiti on a building wall", "COSMETIC"),
        S(("the bus shelter got tagged", "there's graffiti all over the bus stop", "someone spray painted the bus shelter glass"),
          ("bus shelter tagged", "graffiti bus stop"), "Graffiti on a bus shelter", "COSMETIC"),
        S(("someone spray painted the playground slide", "there's graffiti on the playground equipment", "the play structure got tagged"),
          ("graffiti on playground", "playground tagged"), "Graffiti on playground equipment", "COSMETIC"),
        S(("the utility box is covered in tags", "somebody tagged the electrical box", "there's graffiti on the traffic control box"),
          ("tagged utility box", "graffiti on electrical box"), "Graffiti on a utility box", "COSMETIC"),
        S(("the underpass is covered in graffiti", "there's tags all over the bridge underpass", "the tunnel walls got spray painted"),
          ("graffiti in underpass", "underpass tagged"), "Graffiti in an underpass", "COSMETIC"),
        S(("someone tagged the fence", "there's spray paint all over the chain link fence", "the fence got covered in graffiti"),
          ("fence tagged", "graffiti on fence"), "Graffiti on a fence", "COSMETIC"),
        S(("someone spray painted the sidewalk", "there's tags painted on the sidewalk", "somebody painted words all over the pavement"),
          ("graffiti on sidewalk", "sidewalk spray painted"), "Graffiti painted on the sidewalk", "COSMETIC"),
    ],
    "water_sewer": [
        S(("a water main broke", "water is gushing up out of the street", "water is shooting up through the pavement"),
          ("water main break", "water gushing from street"), "Water main break with water gushing from the street", "ACTIVE_DAMAGE"),
        S(("the fire hydrant is leaking", "water keeps trickling out of the hydrant", "the hydrant has been dripping nonstop"),
          ("leaking hydrant", "hydrant dripping"), "Fire hydrant leaking", "QUALITY_OF_LIFE"),
        S(("sewage is backing up into my basement", "the sewer is backing up through our floor drain", "raw sewage is coming up in the basement"),
          ("sewer backup in basement", "sewage in basement"), "Sewer backing up into a home", "HAZARD_SOON"),
        S(("the storm drain is clogged", "the drain at the curb is blocked with leaves", "the catch basin is totally plugged"),
          ("clogged storm drain", "blocked drain"), "Clogged storm drain", "QUALITY_OF_LIFE"),
        S(("we have no water at all", "no water is coming out of the taps", "our water has been off since this morning"),
          ("no water", "water service out"), "Loss of water service", "SERVICE_LOSS"),
        S(("the tap water is coming out brown", "our water is rusty looking", "the water from the faucet is discolored"),
          ("brown tap water", "discolored water"), "Discolored tap water", "SERVICE_LOSS"),
        S(("there's a sewer smell coming from the drain", "it smells like sewage around the manhole", "a rotten sewer odor keeps coming up from the grate"),
          ("sewer smell", "sewage odor"), "Sewer odor from street drain", "QUALITY_OF_LIFE"),
    ],
    "tree_vegetation": [
        S(("a big branch fell on the sidewalk", "a tree limb came down across the sidewalk", "a large branch snapped off onto the sidewalk"),
          ("branch down on sidewalk", "fallen limb"), "Fallen tree limb on the sidewalk", "QUALITY_OF_LIFE"),
        S(("a tree fell across the road", "a whole tree is down blocking the street", "a tree came down and is lying across both lanes"),
          ("tree down across road", "tree blocking street"), "Tree down blocking the roadway", "IMMEDIATE_DANGER"),
        S(("a dead tree is leaning toward the houses", "there's a dead tree that looks ready to fall", "a big dead tree is tilting more every week"),
          ("dead tree leaning", "tree about to fall"), "Dead tree leaning and at risk of falling", "HAZARD_SOON"),
        S(("the hedges are so overgrown you can't use the sidewalk", "bushes have grown over the whole sidewalk", "the shrubs are blocking the sidewalk completely"),
          ("overgrown hedge blocking sidewalk", "bushes over sidewalk"), "Overgrown vegetation blocking the sidewalk", "QUALITY_OF_LIFE"),
        S(("tree branches are covering the stop sign", "bushes block the view at the corner so you can't see cars coming", "the branches hide the stop sign until you're right on it"),
          ("branches covering stop sign", "bushes blocking view"), "Vegetation blocking a traffic sign or sightline", "HAZARD_SOON"),
        S(("the grass in the vacant lot is waist high", "the weeds are like four feet tall", "the empty lot is completely overgrown"),
          ("overgrown lot", "tall weeds"), "Overgrown grass/weeds on a lot", "COSMETIC"),
        S(("a broken branch is hanging by a thread over the road", "a cracked limb is dangling over the street", "there's a snapped branch hanging over the lane"),
          ("hanging branch over road", "dangling limb"), "Broken limb hanging over the roadway", "HAZARD_SOON"),
    ],
    "sidewalk_damage": [
        S(("the sidewalk slab is lifted up", "a chunk of sidewalk is raised like three inches", "one of the sidewalk squares is sticking way up"),
          ("raised sidewalk slab", "lifted sidewalk"), "Raised sidewalk slab", "QUALITY_OF_LIFE"),
        S(("the sidewalk is crumbling", "the concrete on the sidewalk is all broken up", "the sidewalk is falling apart"),
          ("crumbling sidewalk", "broken up sidewalk"), "Crumbling sidewalk", "QUALITY_OF_LIFE"),
        S(("there's a section of sidewalk missing", "a whole piece of sidewalk is gone, it's just dirt", "part of the sidewalk was torn out and never replaced"),
          ("missing sidewalk section", "sidewalk gone"), "Missing sidewalk section", "QUALITY_OF_LIFE"),
        S(("the curb ramp is broken", "the wheelchair ramp at the corner is cracked apart", "the curb cut is busted up"),
          ("broken curb ramp", "busted curb cut"), "Damaged curb ramp", "QUALITY_OF_LIFE"),
        S(("tree roots pushed up the sidewalk", "the roots have busted through the sidewalk", "the sidewalk is heaved up from tree roots"),
          ("roots lifting sidewalk", "root damage sidewalk"), "Sidewalk lifted by tree roots", "QUALITY_OF_LIFE"),
        S(("there's a hole in the sidewalk", "a deep hole opened up in the sidewalk", "part of the sidewalk collapsed into a hole"),
          ("hole in sidewalk", "sidewalk hole"), "Hole in the sidewalk", "HAZARD_SOON"),
        S(("there's a little crack in the sidewalk", "the sidewalk has some hairline cracks", "there's a small crack in the concrete"),
          ("small sidewalk crack", "hairline cracks"), "Minor crack in the sidewalk", "COSMETIC"),
    ],
    "snow_ice": [
        S(("the plow never came through", "the street still hasn't been plowed", "nobody has plowed the road yet"),
          ("street not plowed", "no plow"), "Street not plowed", "QUALITY_OF_LIFE", point=False),
        S(("the sidewalk is a sheet of ice", "the sidewalk is solid ice", "the walkway is super icy"),
          ("icy sidewalk", "sidewalk solid ice"), "Icy sidewalk", "HAZARD_SOON"),
        S(("the plow left a huge snow pile blocking the crosswalk", "the plow pushed a wall of snow across the curb ramp", "there's a plow berm blocking the corner"),
          ("snow pile blocking crosswalk", "plow berm"), "Snow pile left by plow blocking pedestrian access", "QUALITY_OF_LIFE"),
        S(("the bus stop is buried in snow", "nobody cleared the bus stop", "you have to stand in the road to wait for the bus because of the snow"),
          ("bus stop buried in snow", "bus stop not cleared"), "Bus stop not cleared of snow", "QUALITY_OF_LIFE"),
        S(("cars keep sliding on the hill", "the hill is solid ice and cars are sliding", "cars keep sliding through the stop, it's so icy"),
          ("cars sliding on ice", "icy hill"), "Icy road with vehicles sliding", "HAZARD_SOON"),
        S(("the fire hydrant is buried in snow", "you can't even see the hydrant under the snow", "the hydrant is completely snowed in"),
          ("hydrant buried in snow", "snowed in hydrant"), "Fire hydrant buried in snow", "HAZARD_SOON"),
        S(("the road never got salted", "there's no salt on the street at all", "the whole road is glazed and untreated"),
          ("street not salted", "no salt on road"), "Street not salted or treated for ice", "HAZARD_SOON", point=False),
    ],
    "noise": [
        S(("construction is going on at 2am", "they're jackhammering in the middle of the night", "the construction crew starts at 5am with heavy equipment"),
          ("construction noise at night", "jackhammer at 2am"), "Construction noise outside permitted hours", "QUALITY_OF_LIFE"),
        S(("the neighbors blast music until 3am", "there's a loud party every weekend", "someone is playing music so loud the windows shake"),
          ("loud music late night", "loud party"), "Loud music/party late at night", "QUALITY_OF_LIFE"),
        S(("a car alarm has been going off for hours", "a car alarm won't stop", "some car alarm keeps going off all night"),
          ("car alarm going off", "car alarm nonstop"), "Car alarm sounding for an extended period", "QUALITY_OF_LIFE"),
        S(("landscapers run leaf blowers at 6am", "leaf blowers start up super early", "a crew runs leaf blowers before sunrise"),
          ("leaf blowers at 6am", "early leaf blowers"), "Early-morning leaf blower noise", "QUALITY_OF_LIFE"),
        S(("the rooftop unit on the store hums all night", "there's a loud humming from a building's AC", "a big fan on the building drones 24/7"),
          ("humming rooftop unit", "loud AC hum"), "Continuous mechanical humming from a building unit", "QUALITY_OF_LIFE"),
        S(("guys rev motorcycles up and down the street every night", "dirt bikes race up and down the road", "people keep drag racing and revving engines"),
          ("motorcycles revving", "dirt bikes racing"), "Vehicles revving/racing along the street at night", "QUALITY_OF_LIFE", point=False),
        S(("my neighbor runs a saw in his garage sometimes during the day", "there's some noise from a workshop on weekends", "someone does woodworking in the afternoons and it's a bit loud"),
          ("daytime workshop noise", "saw noise afternoons"), "Occasional daytime workshop noise", "COSMETIC"),
    ],
    "abandoned_vehicle": [
        S(("a car hasn't moved in like three weeks", "there's a car that's been parked here for a month", "the same car has been sitting here since last month"),
          ("car not moved in weeks", "car parked a month"), "Vehicle unmoved for an extended period", "COSMETIC"),
        S(("there's a car with flat tires and expired tags", "an old car with no plates is sitting on the street", "a car with four flats has been left here"),
          ("car flat tires expired tags", "car with no plates"), "Vehicle with flat tires and expired or missing plates", "COSMETIC"),
        S(("a car with smashed windows has been sitting there", "an abandoned car with the windows broken out is parked here", "there's a car with busted windows that nobody's touched"),
          ("car with smashed windows", "broken window car"), "Vehicle with broken windows, apparently abandoned", "COSMETIC"),
        S(("someone left a boat trailer on the street", "a trailer has been parked here for weeks", "an empty utility trailer got dumped on the street"),
          ("trailer left on street", "abandoned trailer"), "Trailer left on the street", "COSMETIC"),
        S(("there's a burned out car", "a torched car is sitting there", "somebody left a burnt-up car"),
          ("burned out car", "torched car"), "Burned-out vehicle", "QUALITY_OF_LIFE"),
        S(("an RV has been parked here for weeks", "a camper van hasn't moved in forever", "an old motorhome has been sitting on the street"),
          ("RV parked for weeks", "camper not moving"), "RV/camper parked for an extended period", "COSMETIC"),
        S(("an abandoned car is leaking fluid everywhere", "a junk car is leaking oil onto the street", "some dead car is dripping something all over the road"),
          ("car leaking fluids", "junk car leaking oil"), "Abandoned vehicle leaking fluids", "QUALITY_OF_LIFE"),
    ],
    "traffic_signage": [
        S(("the traffic light is completely out", "the signals are all dark", "the stoplight is dead"),
          ("traffic light out", "signals dark"), "Traffic signal not working", "HAZARD_SOON"),
        S(("the light is stuck on red", "the traffic light is just flashing red", "the signal is stuck and never turns green"),
          ("light stuck on red", "signal stuck"), "Traffic signal malfunctioning", "QUALITY_OF_LIFE"),
        S(("the stop sign got knocked down", "the stop sign is lying on the ground", "somebody hit the stop sign and it's gone"),
          ("stop sign down", "stop sign knocked over"), "Stop sign knocked down", "HAZARD_SOON"),
        S(("the lane lines are totally faded", "you can't see the lane markings anymore", "the road paint is worn off"),
          ("faded lane lines", "no lane markings"), "Faded pavement markings", "COSMETIC", point=False),
        S(("the street name sign is turned the wrong way", "the street sign is bent sideways", "the street sign got twisted around"),
          ("street sign turned", "bent street sign"), "Street name sign bent or turned", "COSMETIC"),
        S(("the walk button doesn't work", "the crosswalk signal never changes to walk", "the pedestrian signal is broken"),
          ("walk button broken", "ped signal broken"), "Pedestrian signal not working", "QUALITY_OF_LIFE"),
        S(("the speed limit sign is missing", "someone took the speed limit sign", "the speed limit sign is gone, just the post is left"),
          ("speed limit sign missing", "no speed sign"), "Speed limit sign missing", "QUALITY_OF_LIFE"),
    ],
    "other": [
        S(("there's a dead deer in the road", "a dead raccoon is lying on the sidewalk", "there's a dead animal by the curb"),
          ("dead animal", "dead deer in road"), "Dead animal needing removal", "QUALITY_OF_LIFE"),
        S(("a bench in the park is broken", "the park bench slats are busted", "one of the benches is snapped in half"),
          ("broken park bench", "busted bench"), "Broken park bench", "COSMETIC"),
        S(("the swing set chain is broken", "the playground slide has a jagged crack", "a bolt is sticking out of the play structure"),
          ("broken swing", "damaged playground"), "Damaged playground equipment", "HAZARD_SOON"),
        S(("a stray dog is running around", "a loose dog keeps chasing people", "there's a dog with no owner roaming around"),
          ("stray dog", "loose dog"), "Loose/stray dog", "QUALITY_OF_LIFE"),
        S(("the park bathroom is always locked", "the public restroom has been closed for weeks", "the restrooms at the park never open"),
          ("park restroom locked", "restroom closed"), "Public restroom closed", "COSMETIC"),
        S(("the drinking fountain doesn't work", "the water fountain at the park is broken", "the drinking fountain just dribbles"),
          ("broken water fountain", "drinking fountain broken"), "Drinking fountain not working", "COSMETIC"),
        S(("someone stapled ads all over the poles", "there's a ton of 'we buy houses' signs on the poles", "people keep posting flyers all over the utility poles"),
          ("signs stapled to poles", "flyers on poles"), "Unauthorized signs posted on utility poles", "COSMETIC"),
    ],
}

CONTEXTS: dict[str, list[Context]] = {
    "pothole_road_damage": [
        C("swerving", ("cars are swerving into the other lane to avoid it", "people keep swerving around it into traffic"), "Vehicles are swerving to avoid it", "HAZARD_SOON", (0, 1, 4, 5, 6)),
        C("vehicle_damage", ("it already popped my tire", "my rim got bent on it yesterday", "it wrecked my tire last night"), "Resident reports vehicle damage from it", "HAZARD_SOON", (0, 1, 4, 5, 6)),
        C("low_traffic", ("it's on a quiet side street", "hardly anyone drives down here", "it's a dead end so there's not much traffic"), "Located on a low-traffic street", None),
        C("cyclists", ("a cyclist almost went down in it", "bikes can't see it at night"), "Cyclists at risk", "HAZARD_SOON", (0, 1, 2, 4, 5, 6)),
        C("long_standing", ("it's been there for months", "it's been like this since spring"), "Reported as long-standing", None, (0, 1, 2, 4, 5, 6)),
    ],
    "streetlight": [
        C("school_route", ("kids walk to school past here in the dark", "it's right by the school crosswalk"), "Pedestrian route used by children", "HAZARD_SOON", (0, 1, 3)),
        C("feels_unsafe", ("it feels sketchy walking here at night", "I don't feel safe walking my dog there after dark"), "Resident reports feeling unsafe in the dark", None, (0, 1, 3)),
        C("sparking", ("the wires are sparking", "I can see sparks coming from it"), "Sparking observed", "IMMEDIATE_DANGER", (4, 5, 6)),
        C("low_use", ("it's a dead end with no houses", "nobody really walks there"), "Low-use area", None),
        C("long_standing", ("it's been like this for weeks", "it's been out since last month"), "Reported as ongoing for weeks", None, (0, 1, 2, 3)),
    ],
    "garbage_illegal_dumping": [
        C("rodents", ("it's attracting rats", "I saw rats in it this morning"), "Rodents observed", "HAZARD_SOON"),
        C("blocking_sidewalk", ("it's blocking the sidewalk so people walk in the street", "wheelchair users can't get past it"), "Blocking the sidewalk", "HAZARD_SOON", (2, 4, 5)),
        C("needles", ("there are needles in the pile", "I saw syringes mixed in"), "Syringes observed in the pile", "HAZARD_SOON", (2, 5, 6)),
        C("days", ("it's been a week", "it's been sitting there for days"), "Reported as ongoing for days", None),
        C("smell", ("it smells awful", "the smell is terrible"), "Strong odor reported", None),
    ],
    "graffiti": [
        C("offensive", ("it's offensive, hateful language", "it's a slur, really offensive stuff", "it's really vulgar and kids see it"), "Content reported as offensive", "QUALITY_OF_LIFE"),
        C("gang", ("it looks like gang tags", "they look like gang signs"), "Resident believes it is gang-related", None),
        C("returned", ("it came back after the city cleaned it", "they painted over it and it's back already"), "Graffiti has reappeared after a previous cleanup", None),
        C("large", ("it's huge, covers the whole thing", "it's a giant piece"), "Large area affected", None),
    ],
    "water_sewer": [
        C("flooding_now", ("it's flooding the street right now", "the water is pouring into people's yards"), "Active flooding reported", "ACTIVE_DAMAGE", (0, 1, 3)),
        C("basement_now", ("it's flooding the basement right now", "it's rising on the floor as we speak"), "Water actively entering the home", "ACTIVE_DAMAGE", (2,)),
        C("icing", ("it's turning to ice on the road", "the runoff is freezing over"), "Water freezing on the roadway", "HAZARD_SOON", (1, 3)),
        C("minor", ("it's just a small trickle", "not a big deal, just a little drip"), "Described as minor", None, (1,)),
        C("neighbors", ("the whole block has it too", "all the neighbors have the same problem"), "Multiple households affected", None, (4, 5, 6)),
    ],
    "tree_vegetation": [
        C("power_lines", ("it's lying on the power lines", "it's resting on the wires"), "Contact with power lines reported", "IMMEDIATE_DANGER", (0, 1, 2, 6)),
        C("peds_in_road", ("people are walking in the street to get around it", "strollers have to go into the road"), "Pedestrians forced into the roadway", "HAZARD_SOON", (0, 3)),
        C("not_blocking", ("it's not blocking anything, just ugly", "it's not in anyone's way"), "Not obstructing", None, (0, 5)),
        C("storm", ("it happened in last night's wind", "it's been like this since the storm"), "Occurred during a recent storm", None),
    ],
    "sidewalk_damage": [
        C("injury", ("my mom tripped on it and fell", "someone tripped there last week and got hurt"), "Resident reports someone was injured", "HAZARD_SOON", (0, 1, 2, 3, 4, 5)),
        C("accessibility", ("wheelchair users can't get past", "I use a walker and can't get through"), "Accessibility blocked", "HAZARD_SOON", (0, 1, 2, 3, 4, 5)),
        C("foot_traffic", ("it's by the senior center, lots of foot traffic", "tons of people walk here every day"), "High foot-traffic area", "HAZARD_SOON", (0, 1, 2, 4, 5)),
        C("minor", ("it's not a big deal", "it's minor but figured I'd report it"), "Described as minor", None),
    ],
    "snow_ice": [
        C("emergency_access", ("an ambulance couldn't get through", "the fire truck got stuck trying to get down here"), "Emergency vehicle access impeded", "IMMEDIATE_DANGER", (0, 4, 6)),
        C("trapped", ("my elderly neighbor can't get out of her house", "older folks on this street can't get out"), "Residents unable to leave their homes", "HAZARD_SOON", (0, 2, 6)),
        C("school", ("kids walk to school on it", "it's right by the elementary school crossing"), "Used by schoolchildren", "HAZARD_SOON", (1, 2, 3)),
        C("days", ("it's been three days since the storm", "it's been days"), "Ongoing since the storm", None),
    ],
    "noise": [
        C("nightly", ("it's every single night", "this happens every night"), "Recurring nightly", None, (0, 1, 2, 4, 5)),
        C("sleep", ("I have a newborn and can't sleep", "my kids can't sleep"), "Resident reports sleep disruption", None, (0, 1, 2, 3, 4, 5)),
        C("once", ("it only happened once", "it's just tonight though"), "Reported as a single occurrence", None),
        C("weekends", ("mostly on weekends", "it's weekends mainly"), "Mostly on weekends", None),
    ],
    "abandoned_vehicle": [
        C("hydrant", ("it's blocking a fire hydrant", "it's parked right in front of the hydrant"), "Blocking a fire hydrant", "HAZARD_SOON"),
        C("driveway", ("it's blocking my driveway", "I can't get out of my driveway"), "Blocking a driveway", "QUALITY_OF_LIFE"),
        C("kids", ("kids are playing in it", "kids climb all over it"), "Children seen playing in or around it", "HAZARD_SOON"),
        C("not_blocking", ("it's not blocking anything", "it's not in the way, just sitting there"), "Not obstructing", None),
    ],
    "traffic_signage": [
        C("busy_moving", ("it's a really busy intersection and cars are just going through", "rush hour traffic is flying through it"), "Heavy traffic moving through", "IMMEDIATE_DANGER", (0, 1, 2)),
        C("near_miss", ("I almost saw a crash", "two cars nearly collided this morning"), "Near-miss collision reported", "HAZARD_SOON", (0, 1, 2, 3, 5, 6)),
        C("school", ("it's by the school crossing", "kids cross here every morning"), "School crossing area", "HAZARD_SOON", (3, 5, 6)),
        C("quiet", ("it's a quiet residential street", "there's barely any traffic"), "Low-traffic area", None),
    ],
    "other": [
        C("smell", ("it's starting to smell", "the smell is bad"), "Odor reported", None, (0,)),
        C("bite", ("it bit someone yesterday", "it nipped a kid on the way to school"), "Resident reports the dog bit someone", "HAZARD_SOON", (3,)),
        C("kids", ("kids play on it all the time", "little kids use it every day"), "Frequently used by children", "HAZARD_SOON", (2,)),
        C("weeks", ("it's been like this for weeks", "it's been weeks"), "Reported as ongoing for weeks", None),
    ],
}

# ---------------------------------------------------------------------------
# Location content
# ---------------------------------------------------------------------------
STREETS = [  # (name, suffix abbreviation, suffix full)
    ("Elm", "St", "Street"), ("Oak", "Ave", "Avenue"), ("Maple", "St", "Street"), ("Pine", "St", "Street"),
    ("Cedar", "Ln", "Lane"), ("Walnut", "St", "Street"), ("Birch", "Rd", "Road"), ("Chestnut", "St", "Street"),
    ("Willow", "Dr", "Drive"), ("Spruce", "St", "Street"), ("Main", "St", "Street"), ("Washington", "Ave", "Avenue"),
    ("Lincoln", "Ave", "Avenue"), ("Jefferson", "St", "Street"), ("Franklin", "Blvd", "Boulevard"), ("Madison", "Ave", "Avenue"),
    ("Park", "Pl", "Place"), ("Lake", "Rd", "Road"), ("Hill", "St", "Street"), ("River", "Rd", "Road"),
    ("Church", "St", "Street"), ("Mill", "Rd", "Road"), ("Spring", "St", "Street"), ("Highland", "Ave", "Avenue"),
    ("Sunset", "Blvd", "Boulevard"), ("Ridge", "Rd", "Road"), ("Forest", "Ave", "Avenue"), ("Meadow", "Ln", "Lane"),
    ("Orchard", "Way", "Way"), ("Cherry", "St", "Street"), ("Harrison", "Ave", "Avenue"), ("Grant", "St", "Street"),
    ("Monroe", "Dr", "Drive"), ("Jackson", "Blvd", "Boulevard"), ("Beacon", "St", "Street"), ("Kingsley", "Ct", "Court"),
    ("1st", "St", "Street"), ("2nd", "Ave", "Avenue"), ("3rd", "St", "Street"), ("4th", "Ave", "Avenue"),
    ("5th", "St", "Street"), ("7th", "Ave", "Avenue"), ("9th", "St", "Street"), ("Old Mill", "Rd", "Road"),
]

LANDMARKS = [
    "Riverside Library", "Lincoln Elementary School", "Harmon Park", "Eastgate Shopping Center",
    "Fire Station 7", "Oakwood Cemetery", "City Hall", "Westside Community Center",
    "Greenfield Middle School", "St. Mary's Church", "Memorial Hospital", "Route 9 Park-and-Ride",
    "Centennial Pool", "Union Station", "Jefferson High School", "Northside Senior Center",
    "Veterans Memorial Park", "Downtown Transit Center",
]
# (resident phrasing prefix, normalized prefix for the ticket's location field)
LANDMARK_RELATIONS = [
    ("near", "near"), ("outside", "outside"), ("behind", "behind"), ("across from", "across from"),
    ("in the parking lot of", "parking lot of"), ("at the entrance to", "entrance of"), ("right next to", "next to"),
]

# (resident phrasings, normalized location). "{a}"/"{b}" = street names.
VAGUE = [
    (("near my house", "by my place", "outside my house"), "near resident's home (exact location unclear)"),
    (("by the park", "next to the park"), "near a park (exact location unclear)"),
    (("down the street from the school", "by the school"), "near a school (exact location unclear)"),
    (("on my block", "on our block"), "resident's block (exact location unclear)"),
    (("at the usual spot", "you know where, same place as last time"), "same place as a previous report (exact location unclear)"),
    (("around the corner from the gas station", "near the gas station"), "near a gas station (exact location unclear)"),
    (("on {a} or maybe {b}, not sure", "either on {a} or {b}, I forget which"), "{A} or {B} (resident unsure)"),
    (("somewhere downtown", "downtown somewhere"), "downtown (exact location unclear)"),
    (("near the big church", "by that church on the hill"), "near a church (exact location unclear)"),
    (("on my way to work", "somewhere on my commute"), "on resident's commute route (exact location unclear)"),
]

# ---------------------------------------------------------------------------
# Surface text content
# ---------------------------------------------------------------------------
GREETINGS = ["Hi,", "Hello,", "Hey", "Good morning,", "Hi there.", "ok so", "Hello 311,", "Hey guys,"]
FORMAL_GREETINGS = ["To whom it may concern,", "Dear City Services,", "Good afternoon,"]
SIGNOFFS = ["Thanks.", "Thank you!", "thx", "- a concerned resident", "Sent from my iPhone", "Appreciate it.", "Please help."]
FORMAL_SIGNOFFS = ["Regards.", "Thank you for your attention.", "Sincerely, a resident."]
FILLERS = [
    "I've lived here 22 years and never had to complain before.",
    "My taxes keep going up and this is what we get.",
    "I already called last week and nobody answered.",
    "Not sure if this is the right place to report this.",
    "My neighbor said she'd call too.",
    "Anyway hope you're having a good day.",
    "The city council never listens to this side of town.",
    "I'm on my lunch break so typing fast.",
    "My dog noticed it before I did lol.",
    "Honestly the whole neighborhood is going downhill.",
    "I saw on Facebook other people are complaining too.",
    "I'm not trying to be a pain about it.",
]
CLAIMS_START = ["URGENT: ", "Not urgent but ", "No rush, but ", "EMERGENCY!!! ", "This is kind of urgent, ", "Low priority but "]
CLAIMS_END = [" Please fix ASAP!!", " Whenever you get a chance.", " This is a real emergency.",
              " Not a big deal though.", " Please send someone today.", " No hurry."]
SECONDARY_JOINERS = ["Also, {x}.", "Oh and {x} too.", "Plus {x}.", "While I'm at it, {x}."]

# Sentence frames. {issue} = issue clause, {locp} = location with preposition,
# {bare} = location text alone. Frames containing a location are used only when
# a location exists; NOLOC frames are used for location_type == "missing".
FRAMES = {
    "F01": "{issue} {locp}.",
    "F02": "{Locp}, {issue}.",
    "F03": "{issue}. it's {locp}.",
    "F04": "I want to report that {issue} {locp}.",
    "F05": "can someone come look at this, {issue} {locp}",
    "F06": "{issue} {locp} and nobody has done anything about it.",
    "F07": "Just letting you know {issue} {locp}.",
    "F08": "{issue} {locp}. can you send someone?",
}
FORMAL_FRAMES = {
    "P01": "I am writing to report that {issue}. Location: {bare}.",
    "P02": "Location: {bare}. Issue: {issue}.",
}
NOLOC_FRAMES = {
    "N01": "{issue}.",
    "N02": "I want to report that {issue}.",
    "N03": "{issue} and nobody has done anything about it.",
    "N04": "can someone look into this, {issue}",
}
TERSE_FRAMES = {"T01": "{short} {bare}", "T02": "{bare} {short}", "T03": "{short} @ {bare}", "T04": "{bare} - {short}"}
TERSE_NOLOC_FRAMES = {"T05": "{short}", "T06": "{short} pls fix"}

STYLES = [("standard", 0.40), ("terse", 0.13), ("verbose", 0.14), ("typo_heavy", 0.11),
          ("slang", 0.10), ("formal", 0.07), ("shouting", 0.05)]

SLANG_SUBS = [(r"\byou\b", "u"), (r"\byour\b", "ur"), (r"\bbecause\b", "cuz"), (r"\bpeople\b", "ppl"),
              (r"\bthough\b", "tho"), (r"\breally\b", "rly"), (r"\bplease\b", "pls"), (r"\bthanks\b", "thx"),
              (r"\bright now\b", "rn"), (r"\bto be honest\b", "tbh"), (r"\bsomeone\b", "sum1"), (r"\bthe\b", "da")]
SLANG_INTERJECTIONS = ["ngl", "smh", "bruh", "lowkey", "fr", "lol", "yall"]

LOC_TOKEN = "⟦LOC⟧"  # placeholder protecting location text from word-level noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cap_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def lower_first(s: str) -> str:
    # Keep acronyms ("RV", "AC") intact.
    if len(s) > 1 and s[1].isupper():
        return s
    return s[:1].lower() + s[1:] if s else s


def street_forms(rng: random.Random, street: tuple[str, str, str], allow_bare: bool = True) -> tuple[str, str]:
    """Return (resident_text, normalized_text) for a street name.

    The normalized form keeps whatever suffix the resident used (abbreviated,
    spelled out, or none) and only fixes casing/trailing periods. It never adds
    a suffix the resident did not write.
    """
    name, abbr, full = street
    choices = ["abbr", "abbr_dot", "full"] + (["none"] if allow_bare else [])
    form = rng.choice(choices)
    if form == "abbr":
        return f"{name} {abbr}", f"{name} {abbr}"
    if form == "abbr_dot":
        return f"{name} {abbr}.", f"{name} {abbr}"
    if form == "full":
        return f"{name} {full}", f"{name} {full}"
    return name, name


def case_variant(rng: random.Random, text: str) -> str:
    r = rng.random()
    if r < 0.45:
        return text.lower()
    if r < 0.50:
        return text.upper()
    return text


def pick_street(rng: random.Random, exclude: tuple = ()) -> tuple[str, str, str]:
    while True:
        s = rng.choice(STREETS)
        if s not in exclude:
            return s


def build_location(rng: random.Random, loc_type: str, scenario: Scenario) -> dict:
    """Return resident-facing and normalized location info for one example."""
    if loc_type == "exact_address":
        r = rng.random()
        number = rng.randint(1, 99) if r < 0.2 else (rng.randint(100, 999) if r < 0.7 else rng.randint(1000, 9999))
        res, norm = street_forms(rng, pick_street(rng))
        bare = case_variant(rng, f"{number} {res}")
        prep = rng.choice(["at", "in front of", "outside", "right by", "at", "on the curb at"])
        address = f"{number} {norm}"
        return {"locp": f"{prep} {LOC_TOKEN}", "bare": LOC_TOKEN, "text": bare,
                "location": address, "address": address}

    if loc_type == "intersection":
        s1 = pick_street(rng)
        s2 = pick_street(rng, exclude=(s1,))
        r1, n1 = street_forms(rng, s1)
        r2, n2 = street_forms(rng, s2)
        pattern = rng.choice(["the corner of {a} and {b}", "{a} and {b}", "{a} & {b}", "{a}/{b}",
                              "where {a} meets {b}", "the intersection of {a} and {b}"])
        text = case_variant(rng, pattern.format(a=r1, b=r2))
        prep = "on" if pattern.startswith("the corner") else ("" if pattern.startswith("where") else "at")
        return {"locp": f"{prep} {LOC_TOKEN}".strip(), "bare": LOC_TOKEN, "text": text,
                "location": f"{n1} and {n2} (intersection)", "address": None}

    if loc_type == "landmark":
        lm = rng.choice(LANDMARKS)
        rel_res, rel_norm = rng.choice(LANDMARK_RELATIONS)
        text = case_variant(rng, lm)
        return {"locp": f"{rel_res} {LOC_TOKEN}", "bare": f"{rel_res} {LOC_TOKEN}", "text": text,
                "location": f"{rel_norm} {lm}", "address": None}

    if loc_type == "block_range":
        s1 = pick_street(rng)
        r1, n1 = street_forms(rng, s1)
        if rng.random() < 0.6:
            hundred = rng.randint(1, 29) * 100
            text = case_variant(rng, f"the {hundred} block of {r1}")
            return {"locp": f"on {LOC_TOKEN}", "bare": LOC_TOKEN, "text": text,
                    "location": f"{hundred} block of {n1}", "address": None}
        s2 = pick_street(rng, exclude=(s1,))
        s3 = pick_street(rng, exclude=(s1, s2))
        r2, n2 = street_forms(rng, s2)
        r3, n3 = street_forms(rng, s3)
        text = case_variant(rng, f"{r1} between {r2} and {r3}")
        return {"locp": f"on {LOC_TOKEN}", "bare": LOC_TOKEN, "text": text,
                "location": f"{n1} between {n2} and {n3}", "address": None}

    if loc_type == "street_only":
        r1, n1 = street_forms(rng, pick_street(rng), allow_bare=False)
        # (with preposition, bare form for "Location: ..." frames). Point issues get
        # phrasings implying one unknown spot; street-wide issues get phrasings that
        # describe the whole street ("somewhere on" would contradict "actionable").
        if scenario.point:
            # no "along {s}": it reads as distributed along the street, contradicting a point issue
            options = [("on {s}", "{s}"), ("somewhere on {s}", "somewhere on {s}"),
                       ("on {s}, not sure of the number", "{s}, not sure of the number")]
        else:
            options = [("on {s}", "{s}"), ("all along {s}", "all along {s}"),
                       ("on {s}, the whole street", "{s}, the whole street"), ("up and down {s}", "up and down {s}")]
        locp, bare = rng.choice(options)
        text = case_variant(rng, r1)
        return {"locp": locp.replace("{s}", LOC_TOKEN), "bare": bare.replace("{s}", LOC_TOKEN),
                "text": text, "location": f"{n1} (no house number given)", "address": None}

    if loc_type == "vague":
        phrasings, norm = rng.choice(VAGUE)
        phrase = rng.choice(phrasings)
        if "{a}" in phrase:
            s1 = pick_street(rng)
            s2 = pick_street(rng, exclude=(s1,))
            r1, n1 = street_forms(rng, s1)
            r2, n2 = street_forms(rng, s2)
            phrase = phrase.format(a=r1, b=r2)
            norm = norm.format(A=n1, B=n2)
        text = case_variant(rng, phrase)
        return {"locp": LOC_TOKEN, "bare": LOC_TOKEN, "text": text, "location": norm, "address": None}

    assert loc_type == "missing"
    return {"locp": "", "bare": "", "text": "", "location": LOCATION_UNSPECIFIED, "address": None}


def needs_clarification(loc_type: str, scenario: Scenario) -> bool:
    """README 3.5: clarify only when a dispatcher could not reasonably find the issue.

    Intersections, landmarks and block ranges are actionable. A street name alone
    is actionable only for street-wide issues (e.g. unplowed street), not for a
    point issue (e.g. one pothole somewhere on a long street).
    """
    if loc_type in ("vague", "missing"):
        return True
    if loc_type == "street_only":
        return scenario.point
    return False


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-z']+")


def add_typos(rng: random.Random, text: str, n: int) -> str:
    spans = [m for m in WORD_RE.finditer(text) if len(m.group()) >= 4]
    if not spans:
        return text
    chosen = sorted(rng.sample(spans, min(n, len(spans))), key=lambda m: m.start(), reverse=True)
    for m in chosen:
        w = m.group()
        i = rng.randint(1, len(w) - 2)
        op = rng.choice(["swap", "drop", "double"])
        if op == "swap":
            w2 = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        elif op == "drop":
            w2 = w[:i] + w[i + 1:]
        else:
            w2 = w[:i] + w[i] + w[i:]
        text = text[:m.start()] + w2 + text[m.end():]
    return text


def add_slang(rng: random.Random, text: str) -> str:
    for pat, rep in SLANG_SUBS:
        if rng.random() < 0.7:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    inter = rng.choice(SLANG_INTERJECTIONS)
    return f"{inter} {text}" if rng.random() < 0.5 else f"{text} {inter}"


def strip_apostrophes(text: str) -> str:
    return text.replace("'", "")


def strip_punct(text: str) -> str:
    return re.sub(r"[.,!?;:]", "", text)


# ---------------------------------------------------------------------------
# Example assembly
# ---------------------------------------------------------------------------
def pick_context(rng: random.Random, category: str, s_idx: int, p: float) -> Context | None:
    if rng.random() >= p:
        return None
    options = [c for c in CONTEXTS[category] if c.only is None or s_idx in c.only]
    return rng.choice(options) if options else None


def render_example(rng: random.Random, slot: dict) -> dict:
    cat, s_idx = slot["category"], slot["scenario_index"]
    scenario = SCENARIOS[cat][s_idx]
    loc_type, style = slot["location_type"], slot["style"]
    loc = build_location(rng, loc_type, scenario)
    noise: list[str] = []

    # --- issue units (primary + optional secondary) ---
    context = None if style == "terse" else pick_context(rng, cat, s_idx, 0.55)
    units = [{"category": cat, "scenario": scenario, "context": context,
              "facts": [scenario.fact] + ([context.fact] if context and context.fact else [])}]
    sec = slot.get("secondary")
    if sec is not None and style != "terse":
        sec_s = SCENARIOS[sec["category"]][sec["scenario_index"]]
        units.append({"category": sec["category"], "scenario": sec_s, "context": None, "facts": [sec_s.fact]})
    elif sec is not None:
        slot["secondary"] = None  # terse complaints stay single-issue

    # --- complaint text ---
    if style == "terse":
        short = rng.choice(scenario.short)
        if loc_type == "missing":
            fid, frame = rng.choice(list(TERSE_NOLOC_FRAMES.items()))
        else:
            fid, frame = rng.choice(list(TERSE_FRAMES.items()))
        body = frame.format(short=short, bare=loc["bare"])
    else:
        issue = rng.choice(scenario.issue)
        if loc_type == "missing":
            fid, frame = rng.choice(list(NOLOC_FRAMES.items()))
        elif style == "formal":
            fid, frame = rng.choice(list(FORMAL_FRAMES.items()))
        else:
            fid, frame = rng.choice(list(FRAMES.items()))
        body = frame.format(issue=issue, locp=loc["locp"], Locp=cap_first(loc["locp"]), bare=loc["bare"])
        body = cap_first(body) if rng.random() < 0.7 else body
        parts = [body]
        if context:
            ctx = rng.choice(context.phrases)
            parts.append(rng.choice([cap_first(ctx) + ".", "And " + ctx + ".", "- " + ctx]))
        if len(units) > 1:
            sec_issue = rng.choice(units[1]["scenario"].issue)
            parts.append(rng.choice(SECONDARY_JOINERS).format(x=sec_issue))
        # Resident urgency claims attach to the complaint itself (before fillers are
        # mixed in) and are drawn independently of the label.
        claim_p = 0.45 if style == "verbose" else 0.15
        if style != "formal" and rng.random() < claim_p:
            if rng.random() < 0.5:
                parts[0] = rng.choice(CLAIMS_START) + lower_first(parts[0])
            else:
                parts[-1] = parts[-1] + rng.choice(CLAIMS_END)
            noise.append("urgency_claim")
        n_fill = {"verbose": rng.choice([1, 2]), "formal": 0}.get(style, 1 if rng.random() < 0.15 else 0)
        fillers = rng.sample(FILLERS, n_fill)
        if fillers:
            if rng.random() < 0.4:
                parts = fillers[:1] + parts + fillers[1:]
            else:
                parts = parts + fillers
            noise.append("irrelevant_detail")
        greeting = None
        if style == "formal":
            greeting = rng.choice(FORMAL_GREETINGS)
            parts = parts + [rng.choice(FORMAL_SIGNOFFS)]
        else:
            if style == "verbose" or rng.random() < 0.25:
                greeting = rng.choice(GREETINGS)
            if style == "verbose" or rng.random() < 0.2:
                parts.append(rng.choice(SIGNOFFS))
        if greeting:
            # "Hello, the light is out" rather than "Hello, The light is out".
            if greeting.endswith(","):
                parts[0] = lower_first(parts[0])
            parts.insert(0, greeting)
        body = " ".join(parts)

    # --- word-level noise (location placeholder is untouched) ---
    if style == "typo_heavy":
        body = add_typos(rng, body, rng.randint(3, 6)); noise.append("typos_heavy")
    elif style in ("standard", "verbose", "slang") and rng.random() < 0.3:
        body = add_typos(rng, body, rng.randint(1, 2)); noise.append("typos_light")
    if style == "slang":
        body = add_slang(rng, body); noise.append("slang")
    if style in ("typo_heavy", "slang", "terse") or rng.random() < 0.2:
        body = strip_apostrophes(body); noise.append("no_apostrophes")
    if style in ("typo_heavy", "terse") and rng.random() < 0.5:
        body = strip_punct(body); noise.append("no_punctuation")
    if style == "shouting":
        body = body.rstrip(".") + "!!!"

    # --- insert location, then casing noise over the whole complaint ---
    complaint = body.replace(LOC_TOKEN, loc["text"])
    complaint = re.sub(r"\s+", " ", complaint).strip()
    complaint = re.sub(r"\.\.(?!\.)", ".", complaint)  # "Elm St." ending a sentence -> single period
    if style == "shouting":
        complaint = complaint.upper(); noise.append("all_caps")
    elif (style in ("slang", "typo_heavy", "terse") and rng.random() < 0.6) or rng.random() < 0.1:
        complaint = complaint.lower(); noise.append("lowercase")

    # --- target ticket (derived only from facts rendered above) ---
    ranks = [max(FACT_RANK[f] for f in u["facts"]) for u in units]
    label_i = max(range(len(units)), key=lambda i: (ranks[i], -i))  # highest urgency; tie -> first mentioned
    label = units[label_i]
    others = [u for i, u in enumerate(units) if i != label_i]

    sent1 = label["scenario"].core
    if label["context"] and label["context"].summary:
        sent1 += "; " + lower_first(label["context"].summary)
    summary = [sent1 + "."]
    for o in others:
        extra = lower_first(o["scenario"].core)
        if o["context"] and o["context"].summary:
            extra += "; " + lower_first(o["context"].summary)
        summary.append(f"Also reports {extra}.")
    clarify = needs_clarification(loc_type, scenario)
    if clarify:
        summary.append(CLARIFICATION_SENTENCE)

    ticket = {
        "category": label["category"],
        "urgency": URGENCY_LEVELS[max(ranks)],
        "location": loc["location"],
        "summary": " ".join(summary),
        "address_or_null": loc["address"],
    }
    assert tuple(ticket) == TICKET_KEYS

    return {
        "family_id": f"{cat}/{s_idx:02d}",
        "secondary_family_id": (f"{slot['secondary']['category']}/{slot['secondary']['scenario_index']:02d}"
                                if slot.get("secondary") else None),
        "category": ticket["category"],
        "urgency": ticket["urgency"],
        "location_type": loc_type,
        "has_exact_address": loc["address"] is not None,
        "needs_clarification": clarify,
        "multi_issue": len(units) > 1,
        "context": context.key if context else None,
        "style": style,
        "frame": fid,
        "noise": noise,
        "complaint": complaint,
        "response": json.dumps(ticket, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Split + allocation
# ---------------------------------------------------------------------------
def spread(total: int, keys: list, rng: random.Random) -> dict:
    """Split `total` as evenly as possible over keys; remainder goes to random keys."""
    base, rem = divmod(total, len(keys))
    extra = set(rng.sample(range(len(keys)), rem))
    return {k: base + (1 if i in extra else 0) for i, k in enumerate(keys)}


def build_dataset(seed: int, data_cfg: dict) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    n_total, n_eval = data_cfg["n_instruction_examples"], data_cfg["n_eval_examples"]
    fams_per_cat = data_cfg["eval_families_per_category"]
    no_addr = data_cfg["no_address_per_condition"]
    multi_rate = data_cfg["multi_issue_rate"]

    # 1. Family-level split, stratified by category.
    split_fams: dict[str, dict[str, list[int]]] = {"train": {}, "eval": {}}
    shuffled: dict[str, list[int]] = {}
    for cat in CATEGORIES:
        idx = list(range(len(SCENARIOS[cat])))
        rng.shuffle(idx)
        shuffled[cat] = idx
        split_fams["eval"][cat] = sorted(idx[:fams_per_cat])
        split_fams["train"][cat] = sorted(idx[fams_per_cat:])

    # 1b. Street-wide constraint (no RNG use): eval must hold at least N street-wide
    # families so "street name only -> actionable" is testable. If the shuffle left
    # too few, the first category (CATEGORIES order) with a street-wide train family
    # swaps it for its last-drawn eval family.
    def n_wide_eval() -> int:
        return sum(not SCENARIOS[c][i].point for c in CATEGORIES for i in split_fams["eval"][c])

    for cat in CATEGORIES:
        if n_wide_eval() >= data_cfg["eval_street_wide_families_min"]:
            break
        wide_train = [i for i in split_fams["train"][cat] if not SCENARIOS[cat][i].point]
        wide_eval = [i for i in split_fams["eval"][cat] if not SCENARIOS[cat][i].point]
        if not wide_train or wide_eval:
            continue
        displaced = shuffled[cat][fams_per_cat - 1]  # last-drawn eval family
        split_fams["eval"][cat] = sorted([i for i in split_fams["eval"][cat] if i != displaced] + [wide_train[0]])
        split_fams["train"][cat] = sorted([i for i in split_fams["train"][cat] if i != wide_train[0]] + [displaced])

    records: dict[str, list[dict]] = {}
    for split, n in (("eval", n_eval), ("train", n_total - n_eval)):
        # 2. Slots per category, then per family (round robin).
        per_cat = spread(n, CATEGORIES, rng)
        slots = []
        for cat in CATEGORIES:
            fams = split_fams[split][cat]
            for j in range(per_cat[cat]):
                slots.append({"category": cat, "scenario_index": fams[j % len(fams)]})

        # 3. Stratified no-address assignment: k conditions spread across categories.
        conds = [c for c in NO_ADDRESS_TYPES for _ in range(no_addr[split])]
        rng.shuffle(conds)
        per_cat_na = spread(len(conds), CATEGORIES, rng)
        chosen = []
        for cat in CATEGORIES:
            cat_slots = [i for i, s in enumerate(slots) if s["category"] == cat]
            chosen += rng.sample(cat_slots, per_cat_na[cat])
        for i in range(len(slots)):
            slots[i]["location_type"] = "exact_address"
        for i, cond in zip(chosen, conds):
            slots[i]["location_type"] = cond

        # 3b. Street-only quota (no RNG use): at least k street_only slots must sit on
        # street-wide families (actionable), while the rest stay on point families
        # (clarification required). Fix-up swaps location types between a point-family
        # street_only slot and a street-wide-family slot, in slot order, so per-condition
        # counts are unchanged.
        def is_wide(s: dict) -> bool:
            return not SCENARIOS[s["category"]][s["scenario_index"]].point

        need = data_cfg["street_only_actionable_min"][split]
        wide_slots = [i for i, s in enumerate(slots) if is_wide(s) and s["location_type"] != "street_only"]
        point_so = [i for i, s in enumerate(slots) if not is_wide(s) and s["location_type"] == "street_only"]
        have = sum(1 for s in slots if is_wide(s) and s["location_type"] == "street_only")
        while have < need and wide_slots and len(point_so) > 1:  # keep >= 1 clarification case
            w, p = wide_slots.pop(0), point_so.pop()
            slots[w]["location_type"], slots[p]["location_type"] = "street_only", slots[w]["location_type"]
            have += 1

        # 4. Style + optional secondary issue (same split, different category).
        styles, weights = zip(*STYLES)
        for s in slots:
            s["style"] = rng.choices(styles, weights)[0]
            s["secondary"] = None
            if rng.random() < multi_rate:
                other = rng.choice([c for c in CATEGORIES if c != s["category"]])
                s["secondary"] = {"category": other,
                                  "scenario_index": rng.choice(split_fams[split][other])}

        # 5. Render; re-draw on exact duplicate complaint (deterministic).
        seen: set[str] = set()
        out = []
        for s in slots:
            for _ in range(20):
                rec = render_example(rng, dict(s))
                if rec["complaint"] not in seen:
                    break
            seen.add(rec["complaint"])
            rec["split"] = split
            out.append(rec)
        rng.shuffle(out)
        for i, rec in enumerate(out, 1):
            rec["id"] = f"cd-{split}-{i:04d}"
        records[split] = out

    split_info = {"eval_families": split_fams["eval"], "train_families": split_fams["train"]}
    return records["train"], records["eval"], split_info


FIELD_ORDER = ["id", "split", "family_id", "secondary_family_id", "category", "urgency", "location_type",
               "has_exact_address", "needs_clarification", "multi_issue", "context", "style", "frame",
               "noise", "complaint", "response"]


def serialize(records: list[dict]) -> str:
    return "".join(json.dumps({k: r[k] for k in FIELD_ORDER}, ensure_ascii=False) + "\n" for r in records)


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    data_cfg = cfg["data"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=data_cfg["seed"])
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for train/eval JSONL (default: paths from configs/adapter_config.json)")
    args = ap.parse_args()

    train, evl, split_info = build_dataset(args.seed, data_cfg)
    if args.out_dir:
        train_path = args.out_dir / "instruction_train.jsonl"
        eval_path = args.out_dir / "instruction_eval.jsonl"
    else:
        train_path = REPO_ROOT / data_cfg["instruction_train_path"]
        eval_path = REPO_ROOT / data_cfg["instruction_eval_path"]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(serialize(train))
    eval_path.write_text(serialize(evl))

    print(f"seed={args.seed}")
    print(f"wrote {len(train)} train -> {train_path}")
    print(f"wrote {len(evl)} eval  -> {eval_path}")
    print("eval families (held out entirely):")
    for cat, idx in split_info["eval_families"].items():
        print(f"  {cat:24s} {[f'{cat}/{i:02d}' for i in idx]}")


if __name__ == "__main__":
    main()
