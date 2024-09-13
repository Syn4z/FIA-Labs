from rules import LEGO_TOURIST_RULES
from production import forward_chain

touristTypes = ["(?x) is a LEGO Technic tourist", "(?x) is a LEGO Creator tourist", "(?x) is a LEGO Star Wars tourist", "(?x) is a LEGO City tourist", "(?x) is a LEGO Adventurers tourist", "(?x) is a Loonie"]

def mainMenu():
    print("\n==========================")
    print("LEGO Tourist Expert System   ")
    print("==========================\n")
    
    print("Welcome! In this system, I'll help identify who your person is.\n")
    
    while True:
        name = input("Enter a name to start: ")
        if name == "":
            print("Name cannot be empty. Please enter a valid name.")
            continue
        else:
            break
    facts = ()
    facts = askMultipleChoiceQuestion(LEGO_TOURIST_RULES, name, facts)
    if facts:
        facts, hypothesis = askRankingQuestion(LEGO_TOURIST_RULES, name, facts)
    else:
        hypothesis = "(?x) is a Loonie"    
    result = askYesNoQuestion(LEGO_TOURIST_RULES, name, facts, hypothesis)
    
    print("\n============================")
    print("Thank you for participating!")
    print("============================\n")
    
    return result

def askRankingQuestion(rules, name, result):
    print(f"\nRank the following activities in order of {name}'s attendance (1 = Most attended, 5 = Least present):")
    preferences = [
        "Engaging with mechanics and engineering models",
        "Participating in modular building activities",
        "Participating in Star Wars-related events and screenings",
        "Engaging with urban planning activities",
        "Participating in adventure-based activities"
    ]
    for i, preference in enumerate(preferences):
        print(f"\t{i + 1}. {preference}")
    
    rankings = {}
    hypothesis = []
    for preference in preferences:
        while True:
            try:
                rank = int(input(f"Enter your rank for {preference} (1-5): "))
                if rank < 1 or rank > 5:
                    raise ValueError("Rank should be between 1 and 5.")
                if rank in rankings.values():
                    raise ValueError("Each rank must be unique.")
                rankings[preference] = rank
                break
            except ValueError as e:
                print(e)
    
    sorted_rankings = sorted(rankings.items(), key=lambda item: item[1], reverse=True)
    fact = []
    for preference, rank in sorted_rankings:
        if preference == "Engaging with mechanics and engineering models":
            fact.append("(?x) engages with mechanics and engineering models")
            hypothesis.append("(?x) is a LEGO Technic tourist")
        elif preference == "Participating in modular building activities":
            fact.append("(?x) participates in modular building activities")
            hypothesis.append("(?x) is a LEGO Creator tourist")
        elif preference == "Participating in Star Wars-related events and screenings":
            fact.append("(?x) participates in Star Wars-related events and screenings")
            hypothesis.append("(?x) is a LEGO Star Wars tourist")
        elif preference == "Engaging with urban planning activities":
            fact.append("(?x) engages with urban planning activities")
            hypothesis.append("(?x) is a LEGO City tourist")
        elif preference == "Participating in adventure-based activities":
            fact.append("(?x) participates in adventure-based activities")
            hypothesis.append("(?x) is a LEGO Adventurers tourist")
    if fact:
        for element in fact:
            y = list(result)
            y.append(element.replace("(?x)", name))
            result = tuple(y)
    result = forward_chain(rules, result)
    print("\nYour ranking:")
    for preference, rank in sorted_rankings:
        print(f"\t{preference}")
    
    return result, hypothesis

def askMultipleChoiceQuestion(rules, name, result):
    print(f"\nWhich of the following activities does {name} participate in?")
    choices = [
        "1. Attend themed events and conventions",
        "2. Purchase LEGO merchandise",
        "3. Engage in building and collecting LEGO sets",
        "4. Take photos of themed LEGO builds",
        "5. None of the above"
    ]
    for choice in choices:
        print("\t" + choice)
    while True:
        try:
            response = int(input(f"Choose the number corresponding to the activity {name} participates in (1-5): "))
            if response < 1 or response > 5:
                raise ValueError
            break
        except ValueError:
            print("Invalid choice. Please enter a number between 1 and 5.")
    if response == 1:
        fact = "(?x) attends themed events and conventions"
    elif response == 2:
        fact = "(?x) purchases merchandise"
    elif response == 3:
        fact = "(?x) engages with building and collecting activities"
    elif response == 4:
        fact = "(?x) takes photos of themed builds"
    else:
        fact = None
    if fact:
        y = list(result)
        y.append(fact.replace("(?x)", name))
        result = tuple(y)
    
    result = forward_chain(rules, result)
    print(f"\nYou selected: {choices[response - 1].split('. ', 1)[1]}")
    
    return result

def askYesNoQuestion(rules, name, result, hypothesis):
    found = False
    if isinstance(hypothesis, list):
        for specific_hypothesis in hypothesis:
            for rule in rules:
                for statement in rule.consequent():
                    if statement == specific_hypothesis:
                        for condition in rule.antecedent():
                            if condition.replace("(?x)", name) not in result:
                                print(condition.replace("(?x)", name) + "?")
                                while True:
                                    response = input("Enter 'yes' or 'no': ").lower()
                                    if response == "yes":
                                        y = list(result)
                                        y.append(condition.replace("(?x)", name))
                                        result = tuple(y)
                                        result = forward_chain(rules, result)
                                        break
                                    elif response == "no":
                                        break
                                    else:
                                        print("Invalid input. Please enter 'yes' or 'no'.")
                                        continue
                                for element in result:
                                    formatted = element.split(" ", 1)
                                    if len(formatted) > 1:
                                        element = f"(?x) {formatted[1]}"
                                    if element in touristTypes:
                                        print("\n " + "--- " + element.replace("(?x)", name) + " ---")
                                        found = True
                                        break
                            else:
                                continue   
                else:
                    continue         
            if found:   
                break
    else:
        for rule in rules:
            for statement in rule.consequent():
                if statement == hypothesis:
                    for condition in rule.antecedent():
                        if condition.replace("(?x)", name) not in result:
                            print(condition.replace("(?x)", name) + "?")
                            while True:
                                response = input("Enter 'yes' or 'no': ").lower()
                                if response == "yes":
                                    y = list(result)
                                    y.append(condition.replace("(?x)", name))
                                    result = tuple(y)
                                    result = forward_chain(rules, result)
                                    break
                                elif response == "no":
                                    break
                                else:
                                    print("Invalid input. Please enter 'yes' or 'no'.")
                                    continue
                            for element in result:
                                formatted = element.split(" ", 1)
                                if len(formatted) > 1:
                                    element = f"(?x) {formatted[1]}"
                                if element in touristTypes:
                                    print("\n " + "--- " + element.replace("(?x)", name) + " ---")
                                    found = True
                                    break
                        else:
                            continue   
                else:
                    continue         
            if found:   
                break

    if not found:
        print(f"\nUnfortunately, I dont have enough information to determine who {name} is.")      
                    
    return result
