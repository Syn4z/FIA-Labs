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
    IF( AND( '(?x) attends themed events or conventions' ),                # L1
        THEN( '(?x) is a tourist' )),

    IF( AND( '(?x) purchases merchandise' ),               # L2
        THEN( '(?x) is a tourist' )),

    IF( AND( '(?x) engages with building or collecting activities' ),      # L3
        THEN( '(?x) is a tourist' )),

    IF( AND( '(?x) participates in building challenges or contests' ),  
        THEN( '(?x) is a tourist' )),
    
    IF( AND( '(?x) is a tourist',                                               # L4                                  
             '(?x) engages with mechanics or engineering models',               
             '(?x) purchases complex or functional LEGO sets' ), 
        THEN( '(?x) is a LEGO Technic tourist' )),
    
    IF( AND( '(?x) is a tourist',                                               # L5                          
             '(?x) participates in custom or modular building activities',      
             '(?x) shows interest in the aesthetic and creative aspects of LEGO builds' ),  
        THEN( '(?x) is a LEGO Creator tourist' )),

    IF( AND( '(?x) is a tourist',                                               # L6                  
             '(?x) is focused on Star Wars-themed sets',                        
             '(?x) participates in Star Wars-related events or screenings' ),     
        THEN( '(?x) is a LEGO Star Wars tourist' )),
    
    IF( AND( '(?x) is a tourist',                                               # L7                  
             '(?x) engages with city-building or urban planning activities',    
             '(?x) shows interest in realistic cityscape or civic buildings' ),  
        THEN( '(?x) is a LEGO City tourist' )),

    IF( AND( '(?x) is a tourist',                                               # L8              
             '(?x) seeks adventure or exploration-themed LEGO sets',            
             '(?x) participates in adventure-based activities or simulations' ),  
        THEN( '(?x) is a LEGO Adventurers tourist' )),
)

LEGO_TOURIST_DATA = (
    'joe purchases LEGO sets or related merchandise',
    'joe engages with mechanics or engineering models',
    'joe purchases complex or functional LEGO sets',
    # 'joe has screwdriver',
    'sue purchases LEGO sets or related merchandise',
    'sue participates in custom or modular building activities',
    'sue shows interest in the aesthetic and creative aspects of LEGO builds',
    # 'sue has sketchbook',
)

