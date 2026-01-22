"""
Morphene Instrument Definitions

50 neural audio instruments built on RAVE, each combining a musical source
with a non-musical source for real-time timbre morphing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Tier(Enum):
    """Instrument tiers based on priority and complexity."""
    FLAGSHIP = 1  # 10 instruments - highest priority
    STRONG = 2    # 20 instruments - solid pairings
    EXPERIMENTAL = 3  # 20 instruments - higher experimentation tolerance


@dataclass
class Instrument:
    """Definition of a Morphene instrument."""
    number: int
    name: str
    musical_source: str
    nonmusical_source: str
    tier: Tier
    musical_source_description: str = ""
    nonmusical_source_description: str = ""
    notes: str = ""

    @property
    def folder_name(self) -> str:
        """Return normalized folder name for this instrument."""
        return self.name.lower()

    @property
    def display_name(self) -> str:
        """Return formatted display name."""
        return f"{self.number:02d}. {self.name}"


# Tier 1 - Flagship (10 instruments)
TIER1_FLAGSHIP = [
    Instrument(
        number=1,
        name="Traffijuno",
        musical_source="Juno sawtooth pad",
        nonmusical_source="Street traffic",
        tier=Tier.FLAGSHIP,
        musical_source_description="Roland Juno-60/106 sawtooth pad with slow attack, medium release, chorus enabled",
        nonmusical_source_description="Urban street traffic - cars passing, engines, horns, tire noise",
    ),
    Instrument(
        number=5,
        name="Oceancello",
        musical_source="Cello",
        nonmusical_source="Ocean waves",
        tier=Tier.FLAGSHIP,
        musical_source_description="Solo cello - sustained notes, legato phrases, various dynamics and vibrato",
        nonmusical_source_description="Ocean waves - breaking on shore, underwater rumble, spray",
    ),
    Instrument(
        number=10,
        name="Pyroviolin",
        musical_source="Violin",
        nonmusical_source="Crackling fire",
        tier=Tier.FLAGSHIP,
        musical_source_description="Solo violin - full range, various bowing techniques, pizzicato, harmonics",
        nonmusical_source_description="Crackling fire - wood pops, embers, sustained flame roar",
        notes="POC instrument - first to be trained and validated",
    ),
    Instrument(
        number=18,
        name="Thunderslide",
        musical_source="Slide guitar",
        nonmusical_source="Thunderstorm",
        tier=Tier.FLAGSHIP,
        musical_source_description="Slide guitar - lap steel and bottleneck, blues and ambient styles",
        nonmusical_source_description="Thunderstorm - distant and close thunder, rain, wind",
    ),
    Instrument(
        number=20,
        name="Fallflute",
        musical_source="Alto flute",
        nonmusical_source="Waterfall",
        tier=Tier.FLAGSHIP,
        musical_source_description="Alto flute - breathy tones, legato, flutter tongue, key clicks",
        nonmusical_source_description="Waterfall - varying distances, spray, pool churning",
    ),
    Instrument(
        number=26,
        name="Caférhodes",
        musical_source="Rhodes piano",
        nonmusical_source="Coffee shop ambience",
        tier=Tier.FLAGSHIP,
        musical_source_description="Fender Rhodes - clean and driven tones, various velocities, tremolo",
        nonmusical_source_description="Coffee shop - espresso machines, cups clinking, murmured conversation",
    ),
    Instrument(
        number=30,
        name="Ventichoir",
        musical_source="Mellotron choir",
        nonmusical_source="Hospital ventilator",
        tier=Tier.FLAGSHIP,
        musical_source_description="Mellotron M400 choir patches - various vowels and sustains",
        nonmusical_source_description="Hospital ventilator - rhythmic pumping, hissing, mechanical cycling",
    ),
    Instrument(
        number=41,
        name="Insectlan",
        musical_source="Gamelan",
        nonmusical_source="Jungle insects at night",
        tier=Tier.FLAGSHIP,
        musical_source_description="Javanese gamelan - metallophones, gongs, various ensemble textures",
        nonmusical_source_description="Nocturnal jungle insects - cicadas, crickets, katydids, layered chorus",
    ),
    Instrument(
        number=47,
        name="Frogbira",
        musical_source="Mbira",
        nonmusical_source="Rainforest frogs",
        tier=Tier.FLAGSHIP,
        musical_source_description="Mbira dzavadzimu - traditional patterns, bottle resonator variations",
        nonmusical_source_description="Rainforest frog chorus - multiple species, varying density",
    ),
    Instrument(
        number=50,
        name="Demoliano",
        musical_source="Prepared piano",
        nonmusical_source="Demolition debris",
        tier=Tier.FLAGSHIP,
        musical_source_description="Prepared piano - muted strings, metallic preparations, extended techniques",
        nonmusical_source_description="Demolition site - falling debris, breaking concrete, metal impacts",
    ),
]

# Tier 2 - Strong (20 instruments)
TIER2_STRONG = [
    Instrument(
        number=3,
        name="Raininet",
        musical_source="Clarinet",
        nonmusical_source="Rain on tin roof",
        tier=Tier.STRONG,
        musical_source_description="Bb clarinet - full range, legato, staccato, multiphonics",
        nonmusical_source_description="Rain on corrugated tin roof - varying intensity, drips, streams",
    ),
    Instrument(
        number=4,
        name="Windshaku",
        musical_source="Shakuhachi",
        nonmusical_source="Wind through trees",
        tier=Tier.STRONG,
        musical_source_description="Shakuhachi flute - traditional honkyoku techniques, breath tones",
        nonmusical_source_description="Wind through various tree types - leaves rustling, branches creaking",
    ),
    Instrument(
        number=7,
        name="Boilbowl",
        musical_source="Singing bowl",
        nonmusical_source="Boiling water",
        tier=Tier.STRONG,
        musical_source_description="Tibetan singing bowls - various sizes, struck and rubbed, overtones",
        nonmusical_source_description="Boiling water - bubbling, rolling boil, steam release",
    ),
    Instrument(
        number=9,
        name="Fridgebass",
        musical_source="Fretless bass",
        nonmusical_source="Refrigerator drone",
        tier=Tier.STRONG,
        musical_source_description="Fretless electric bass - slides, sustained notes, harmonics",
        nonmusical_source_description="Refrigerator compressor - low drone, cycling, hum variations",
    ),
    Instrument(
        number=13,
        name="Subwaymonca",
        musical_source="Harmonica",
        nonmusical_source="Subway train interior",
        tier=Tier.STRONG,
        musical_source_description="Diatonic and chromatic harmonica - blues bends, chords, single notes",
        nonmusical_source_description="Subway car interior - wheel rumble, announcements, door chimes",
    ),
    Instrument(
        number=17,
        name="Cicaderhu",
        musical_source="Erhu",
        nonmusical_source="Cicadas",
        tier=Tier.STRONG,
        musical_source_description="Erhu (Chinese violin) - vibrato, glissando, traditional ornaments",
        nonmusical_source_description="Cicada chorus - summer peak, individual calls, mass drone",
    ),
    Instrument(
        number=21,
        name="Staticsax",
        musical_source="Soprano sax",
        nonmusical_source="Shortwave radio static",
        tier=Tier.STRONG,
        musical_source_description="Soprano saxophone - classical and jazz articulations, overtones",
        nonmusical_source_description="Shortwave radio - static, tuning between stations, interference",
    ),
    Instrument(
        number=23,
        name="Sizzlebow",
        musical_source="Bowed cymbal",
        nonmusical_source="Sizzling oil",
        tier=Tier.STRONG,
        musical_source_description="Bowed cymbals - various sizes, pressure variations, harmonics",
        nonmusical_source_description="Oil sizzling - frying pan, varying temperatures, food dropping in",
    ),
    Instrument(
        number=27,
        name="Stationstrings",
        musical_source="String quartet",
        nonmusical_source="Train station announcements",
        tier=Tier.STRONG,
        musical_source_description="String quartet - ensemble textures, unison to complex harmonies",
        nonmusical_source_description="Train station - PA announcements, platform ambience, train arrivals",
    ),
    Instrument(
        number=29,
        name="Stepsorgan",
        musical_source="Pipe organ",
        nonmusical_source="Cathedral footsteps",
        tier=Tier.STRONG,
        musical_source_description="Pipe organ - various stops, pedal notes, full ensemble",
        nonmusical_source_description="Footsteps in cathedral - stone floors, various paces, reverberant space",
    ),
    Instrument(
        number=32,
        name="Playgroundharp",
        musical_source="Harp arpeggios",
        nonmusical_source="Children's playground",
        tier=Tier.STRONG,
        musical_source_description="Concert harp - arpeggiated patterns, glissandos, harmonics",
        nonmusical_source_description="Playground - children playing, swings, slides, distant shouts",
    ),
    Instrument(
        number=33,
        name="Hailrimba",
        musical_source="Marimba",
        nonmusical_source="Hailstorm",
        tier=Tier.STRONG,
        musical_source_description="Marimba - full range, rolls, melodic passages, mallet variations",
        nonmusical_source_description="Hailstorm - various hail sizes, on different surfaces, with wind",
    ),
    Instrument(
        number=35,
        name="Bambooto",
        musical_source="Koto",
        nonmusical_source="Bamboo forest wind",
        tier=Tier.STRONG,
        musical_source_description="Japanese koto - traditional techniques, bends, harmonics",
        nonmusical_source_description="Wind through bamboo grove - stalks creaking, leaves rustling",
    ),
    Instrument(
        number=37,
        name="Clocklesta",
        musical_source="Celesta",
        nonmusical_source="Clock mechanisms",
        tier=Tier.STRONG,
        musical_source_description="Celesta - delicate passages, full range, varying velocities",
        nonmusical_source_description="Clock mechanisms - ticking, chiming, gear movements, pendulums",
    ),
    Instrument(
        number=39,
        name="Hoofschord",
        musical_source="Harpsichord",
        nonmusical_source="Horse hooves on cobblestone",
        tier=Tier.STRONG,
        musical_source_description="Harpsichord - baroque passages, ornamentation, multiple registers",
        nonmusical_source_description="Horse hooves on cobblestone - walking, trotting, multiple horses",
    ),
    Instrument(
        number=42,
        name="Crumplebox",
        musical_source="Music box",
        nonmusical_source="Paper crumpling",
        tier=Tier.STRONG,
        musical_source_description="Music box mechanisms - various melodies, winding down, pristine",
        nonmusical_source_description="Paper crumpling - various paper types, crinkles, tears, balls",
    ),
    Instrument(
        number=44,
        name="Flagtoor",
        musical_source="Santoor",
        nonmusical_source="Prayer flag flutter",
        tier=Tier.STRONG,
        musical_source_description="Santoor (hammered dulcimer) - traditional Persian/Indian techniques",
        nonmusical_source_description="Prayer flags fluttering - wind gusts, fabric snapping, rope creaking",
    ),
    Instrument(
        number=45,
        name="Clinkspiel",
        musical_source="Glockenspiel",
        nonmusical_source="Glass bottles clinking",
        tier=Tier.STRONG,
        musical_source_description="Glockenspiel - bright melodic passages, various mallet types",
        nonmusical_source_description="Glass bottles - clinking, rolling, various sizes and fill levels",
    ),
    Instrument(
        number=48,
        name="Projectorgan",
        musical_source="Pump organ",
        nonmusical_source="Film projector",
        tier=Tier.STRONG,
        musical_source_description="Reed pump organ - hymn-like passages, stops variations, bellows",
        nonmusical_source_description="Film projector - sprocket sounds, motor hum, reel changes",
    ),
    Instrument(
        number=49,
        name="Arcadiano",
        musical_source="Toy piano",
        nonmusical_source="Arcade machines",
        tier=Tier.STRONG,
        musical_source_description="Toy piano - tinny timbre, various articulations, full range",
        nonmusical_source_description="Arcade machines - game sounds, coin drops, attract modes, crowds",
    ),
]

# Tier 3 - Experimental (20 instruments)
TIER3_EXPERIMENTAL = [
    Instrument(
        number=2,
        name="Washmoog",
        musical_source="Moog bass",
        nonmusical_source="Washing machine cycle",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Minimoog bass - classic patches, filter sweeps, portamento",
        nonmusical_source_description="Washing machine - full cycle, agitation, spin, water filling",
    ),
    Instrument(
        number=6,
        name="Humeremin",
        musical_source="Theremin",
        nonmusical_source="Fluorescent light hum",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Theremin - vibrato, glissando, various playing styles",
        nonmusical_source_description="Fluorescent light hum - 60Hz buzz, flicker, ballast noise",
    ),
    Instrument(
        number=8,
        name="Birdboe",
        musical_source="Oboe",
        nonmusical_source="Birdsong chatter",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Oboe - full range, various articulations, double tonguing",
        nonmusical_source_description="Birdsong - multiple species, dawn chorus, calls and responses",
    ),
    Instrument(
        number=11,
        name="Cabinduk",
        musical_source="Duduk",
        nonmusical_source="Airplane cabin noise",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Armenian duduk - sustained notes, ornaments, traditional melodies",
        nonmusical_source_description="Airplane cabin - engine drone, air circulation, announcements",
    ),
    Instrument(
        number=12,
        name="Typerecord",
        musical_source="Recorder",
        nonmusical_source="Mechanical keyboard typing",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Recorder family - soprano to bass, baroque articulations",
        nonmusical_source_description="Mechanical keyboard - various switch types, typing patterns",
    ),
    Instrument(
        number=14,
        name="Constructbone",
        musical_source="Trombone",
        nonmusical_source="Construction site",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Trombone - glissando, muted, various articulations",
        nonmusical_source_description="Construction site - hammering, machinery, shouting, vehicles",
    ),
    Instrument(
        number=15,
        name="Murmurwhistle",
        musical_source="Human whistle",
        nonmusical_source="Crowd murmur",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Human whistling - melodic, sustained, vibrato, bird-like",
        nonmusical_source_description="Crowd murmur - various sizes, indoor/outdoor, anticipation",
    ),
    Instrument(
        number=16,
        name="Dryerlimba",
        musical_source="Kalimba",
        nonmusical_source="Coins in dryer",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Kalimba/thumb piano - traditional patterns, various tunings",
        nonmusical_source_description="Coins tumbling in dryer - rhythmic tumbling, impacts, slides",
    ),
    Instrument(
        number=19,
        name="Factoryphone",
        musical_source="Bowed vibraphone",
        nonmusical_source="Factory machinery",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Bowed vibraphone - sustained tones, motor on/off, harmonics",
        nonmusical_source_description="Factory machinery - rhythmic machines, conveyors, presses",
    ),
    Instrument(
        number=22,
        name="Highwayhorn",
        musical_source="French horn",
        nonmusical_source="Highway at distance",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="French horn - stopped and open, various dynamics, hunting calls",
        nonmusical_source_description="Distant highway - constant drone, occasional vehicles, doppler",
    ),
    Instrument(
        number=24,
        name="Icearmonica",
        musical_source="Glass armonica",
        nonmusical_source="Ice cracking",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Glass armonica - ethereal sustained tones, chords, melody",
        nonmusical_source_description="Ice cracking - lake ice, glacial calving, ice cubes",
    ),
    Instrument(
        number=25,
        name="Transformondes",
        musical_source="Ondes Martenot",
        nonmusical_source="Electrical transformer hum",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Ondes Martenot - ribbon control, palme speaker, various timbres",
        nonmusical_source_description="Electrical transformer hum - 60Hz and harmonics, varying loads",
    ),
    Instrument(
        number=28,
        name="Laundrylitzer",
        musical_source="Wurlitzer",
        nonmusical_source="Laundromat",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Wurlitzer electric piano - warm tones, tremolo, various velocities",
        nonmusical_source_description="Laundromat ambience - multiple machines, coin drops, folding",
    ),
    Instrument(
        number=31,
        name="Barkguitar",
        musical_source="Acoustic guitar chords",
        nonmusical_source="Dog barking",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Acoustic guitar - strummed chords, fingerpicking, harmonics",
        nonmusical_source_description="Dog barking - various breeds, distances, excitement levels",
    ),
    Instrument(
        number=34,
        name="Cashdulcimer",
        musical_source="Hammered dulcimer",
        nonmusical_source="Cash register and coins",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Hammered dulcimer - traditional patterns, arpeggios, sustain",
        nonmusical_source_description="Cash register - drawer opening, coins counting, receipt printing",
    ),
    Instrument(
        number=36,
        name="Marketdrum",
        musical_source="Steel drum",
        nonmusical_source="Market haggling voices",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Steel pan/drum - Caribbean melodies, various registers",
        nonmusical_source_description="Market haggling - multiple languages, bargaining, calling out",
    ),
    Instrument(
        number=38,
        name="Tangoneon",
        musical_source="Bandoneon",
        nonmusical_source="Tango footsteps on wood",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Bandoneon - tango phrasing, bellows accents, full range",
        nonmusical_source_description="Tango footsteps - heels on wooden floor, various tempos",
    ),
    Instrument(
        number=40,
        name="Bowlophone",
        musical_source="Vibraphone chords",
        nonmusical_source="Bowling alley",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Vibraphone - chord voicings, motor on/off, mallet dampening",
        nonmusical_source_description="Bowling alley - balls rolling, pins falling, reset mechanisms",
    ),
    Instrument(
        number=43,
        name="Mowerharp",
        musical_source="Autoharp",
        nonmusical_source="Lawnmower",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Autoharp - strummed chords, various chord bars, arpeggios",
        nonmusical_source_description="Lawnmower - gas engine, starting, idling, cutting grass",
    ),
    Instrument(
        number=46,
        name="Brakecordion",
        musical_source="Accordion",
        nonmusical_source="Bus air brakes",
        tier=Tier.EXPERIMENTAL,
        musical_source_description="Accordion - bellows work, bass buttons, various styles",
        nonmusical_source_description="Bus air brakes - hissing, pressure release, door mechanisms",
    ),
]

# Combined list of all instruments
ALL_INSTRUMENTS = TIER1_FLAGSHIP + TIER2_STRONG + TIER3_EXPERIMENTAL

# Lookup dictionaries
INSTRUMENTS_BY_NUMBER = {i.number: i for i in ALL_INSTRUMENTS}
INSTRUMENTS_BY_NAME = {i.name.lower(): i for i in ALL_INSTRUMENTS}
INSTRUMENTS_BY_TIER = {
    Tier.FLAGSHIP: TIER1_FLAGSHIP,
    Tier.STRONG: TIER2_STRONG,
    Tier.EXPERIMENTAL: TIER3_EXPERIMENTAL,
}


def get_instrument(identifier: str | int) -> Optional[Instrument]:
    """
    Get an instrument by number or name.

    Args:
        identifier: Instrument number (int) or name (str)

    Returns:
        Instrument if found, None otherwise
    """
    if isinstance(identifier, int):
        return INSTRUMENTS_BY_NUMBER.get(identifier)
    return INSTRUMENTS_BY_NAME.get(identifier.lower())


def get_instruments_by_tier(tier: Tier) -> list[Instrument]:
    """Get all instruments in a given tier."""
    return INSTRUMENTS_BY_TIER.get(tier, [])


def get_poc_instrument() -> Instrument:
    """Get the POC instrument (Pyroviolin)."""
    return INSTRUMENTS_BY_NUMBER[10]


def list_instruments(tier: Optional[Tier] = None) -> None:
    """Print a formatted list of instruments."""
    instruments = get_instruments_by_tier(tier) if tier else ALL_INSTRUMENTS

    current_tier = None
    for inst in sorted(instruments, key=lambda x: (x.tier.value, x.number)):
        if inst.tier != current_tier:
            current_tier = inst.tier
            tier_names = {
                Tier.FLAGSHIP: "TIER 1 - FLAGSHIP",
                Tier.STRONG: "TIER 2 - STRONG",
                Tier.EXPERIMENTAL: "TIER 3 - EXPERIMENTAL",
            }
            print(f"\n=== {tier_names[current_tier]} ===\n")

        print(f"  {inst.display_name}")
        print(f"    Musical: {inst.musical_source}")
        print(f"    Non-musical: {inst.nonmusical_source}")
        print()


if __name__ == "__main__":
    # Print all instruments when run directly
    print("MORPHENE INSTRUMENT COLLECTION")
    print("=" * 50)
    print(f"Total instruments: {len(ALL_INSTRUMENTS)}")
    print(f"  Tier 1 (Flagship): {len(TIER1_FLAGSHIP)}")
    print(f"  Tier 2 (Strong): {len(TIER2_STRONG)}")
    print(f"  Tier 3 (Experimental): {len(TIER3_EXPERIMENTAL)}")

    list_instruments()
