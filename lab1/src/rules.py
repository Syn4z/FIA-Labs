from production import IF, AND, THEN, OR, DELETE, NOT, FAIL


# Tourists types
"""
LunaCity is a LEGO city where tourists can admire all kind of LEGO sets and collections in one place.
1. LEGO Technic (tan skin, tall, eyeglasses, screwdriver, likes vehicles, )
2. LEGO Creator (brown skin, average height, sunglasses, sketchbook, likes art, )
3. LEGO Star Wars (light skin, short, sunglasses, watch, likes movies, )
4. LEGO City (light skin, short, baseball cap, camera, likes architecture, )
5. LEGO Adventurers (dark skin, average height, hat, compass, likes museums, )

6. LunaCity citizen(NOT a tourist)(normal skin, )
"""

#TO DO: Define the LEGO_TOURIST_RULES
LEGO_TOURIST_RULES = (
    IF( AND( '(?x) has tan skin',       # L1
             '(?x) has brown skin', 
             '(?x) has light skin',
             '(?x) has dark skin'),         
        THEN( '(?x) is a tourist' )), 
)

LEGO_TOURIST_DATA = (
    'joe has tan skin',
    'joe is tall',
    'joe has eyeglasses',
    'joe has screwdriver',
    'sue has brown skin',
    'sue is average height',
    'sue has sunglasses',
    'sue has sketchbook',
)

