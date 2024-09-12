from rules import LEGO_TOURIST_RULES
from production import forward_chain

def mainMenu():
    print("\n==============================")
    print("   LEGO Tourist Expert System   ")
    print("==============================\n")
    
    print("Welcome! In this system, I'll help identify who you are.\n")
    
    while True:
        name = input("Enter your name to start: ")
        if name == "":
            print("Name cannot be empty. Please enter a valid name.")
            continue
        else:
            break
    
    result = askYesNoQuestion(LEGO_TOURIST_RULES, name)
    
    print("\n======================================")
    print("Thank you for participating, {}!".format(name))
    print("======================================\n")
    
    return result

def askYesNoQuestion(rules, name):
    result = ()
    touristTypes = ["(?x) is a LEGO Technic tourist", "(?x) is a LEGO Creator tourist", "(?x) is a LEGO Star Wars tourist", "(?x) is a LEGO City tourist", "(?x) is a LEGO Adventurers tourist", "(?x) is a looney"]
    found = False
    for rule in rules:
        for expr in rule.antecedent():   
            if rule.consequent()[0].replace("(?x)", name) not in result:    
                if expr.replace("(?x)", name) in result:
                    continue
                elif expr == "(?x) is a tourist" and result == ():   
                    break
                print(expr.replace("(?x)", name) + "?")
                while True:
                    response = input("Enter 'yes' or 'no': ").lower()
                    if response == "yes":
                        y = list(result)
                        y.append(expr.replace("(?x)", name))
                        result = tuple(y)
                        result = forward_chain(rules, result)
                        break  
                    elif response == "no":
                        break  
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")
                        continue  
                for i in result:
                    formatted = i.split(" ", 1)
                    if len(formatted) > 1:
                        i = f"(?x) {formatted[1]}"
                    if i in touristTypes:
                        print("\n " + i.replace("(?x)", name)) 
                        found = True
                        break   
            else:
                continue     
        if found:   
            break
    if not found:
        print("\nUnfortunately, I dont have enough information to determine who are you.")      
                    
    return result
