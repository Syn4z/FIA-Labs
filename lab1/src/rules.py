from production import IF, AND, THEN, OR, DELETE, NOT, FAIL


# Tourists types
"""
Luna-City is a LEGO city where tourists can admire all kind of LEGO sets and collections in one place.
1. LEGO Technic tourist
2. LEGO Creator tourist
3. LEGO Star Wars tourist
4. LEGO City tourist
5. LEGO Adventurers tourist

6. Luna-City citizen(NOT a tourist)
"""

LEGO_TOURIST_RULES = (
    IF( AND( '(?x) attends themed events and conventions' ),                    # L1
        THEN( '(?x) is a tourist' )),

    IF( AND( '(?x) purchases merchandise' ),                                    # L2
        THEN( '(?x) is a tourist' )),

    IF( AND( '(?x) engages with building and collecting activities' ),          # L3
        THEN( '(?x) is a tourist' )),

    IF( AND( '(?x) takes photos of themed builds' ),  
        THEN( '(?x) is a tourist' )),
    
    IF( AND( '(?x) is a tourist',                                               # L4                                  
             '(?x) engages with mechanics and engineering models',               
             '(?x) purchases complex and functional LEGO sets' ), 
        THEN( '(?x) is a LEGO Technic tourist' )),
    
    IF( AND( '(?x) is a tourist',                                               # L5                          
             '(?x) participates in modular building activities',      
             '(?x) shows interest in the creative aspects of LEGO builds' ),  
        THEN( '(?x) is a LEGO Creator tourist' )),

    IF( AND( '(?x) is a tourist',                                               # L6                  
             '(?x) is focused on Star Wars-themed sets',                        
             '(?x) participates in Star Wars-related events and screenings' ),     
        THEN( '(?x) is a LEGO Star Wars tourist' )),
    
    IF( AND( '(?x) is a tourist',                                               # L7                  
             '(?x) engages with urban planning activities',    
             '(?x) shows interest in realistic cityscape and civic buildings' ),  
        THEN( '(?x) is a LEGO City tourist' )),

    IF( AND( '(?x) is a tourist',                                               # L8              
             '(?x) seeks exploration-themed LEGO sets',            
             '(?x) participates in adventure-based activities' ),  
        THEN( '(?x) is a LEGO Adventurers tourist' )),

    IF( AND( '(?x) visits less-known or hidden LEGO locations',                 # L9
             '(?x) can give directions to local LEGO attractions' ),                                       
        THEN( '(?x) is a Loonie' ))      
)

LEGO_TOURIST_DATA = (
    'joe attends themed events and conventions',
    'joe engages with mechanics and engineering models',
    'joe purchases complex and functional LEGO sets',
)
