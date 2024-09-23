from rules import LEGO_TOURIST_RULES
from production import forward_chain, backward_chain

touristTypes = ["(?x) is a LEGO Technic tourist", "(?x) is a LEGO Creator tourist", "(?x) is a LEGO Star Wars tourist", "(?x) is a LEGO City tourist", "(?x) is a LEGO Adventurers tourist", "(?x) is a Loonie"]

def mainMenu():
    """
    Displays the main menu of the Expert System and manages user input for 
    either hypothesis-driven identification or answering questions to identify a LEGO tourist.

    Users can choose to either input a hypothesis or answer a series of multiple-choice and yes/no questions
    to identify a LEGO tourist.
    
    Returns:
        result: The outcome of the expert system's evaluation based on the input (a conclusion about the tourist type).
    """
    print("\n==========================")
    print("LEGO Tourist Expert System")
    print("==========================\n")
    
    print("Welcome! In this system, I'll help identify who your person is.\n")

    # Continuously prompt user until they provide valid input (1 or 2)
    while True:
        inputChoice = input("Press 1 to give a hypothesis or Press 2 to answer questions: ")
        if inputChoice == "1" or inputChoice == "2":
            break
        else:
            print("Invalid input. Please enter '1' or '2'.")

    # Handle user selection for hypothesis or question-answering
    if inputChoice == "1":
        # Hypothesis-driven approach: prompt user to enter a hypothesis
        hypothesis = input("Enter a hypothesis in the format \"{name} is a LEGO Technic tourist\": ")
        result, humanReadable = backward_chain(LEGO_TOURIST_RULES, hypothesis)
        print(humanReadable)
    elif inputChoice == "2":
        # Question-driven approach: ask the user for a name and proceed with multiple-choice questions
        while True:
            name = input("Enter a name to start: ")
            if name == "":
                print("Name cannot be empty. Please enter a valid name.")
                continue
            else:
                break
        # Initialize an empty set of facts to track user responses
        facts = ()
        # Ask multiple-choice questions to gather initial facts
        facts = askMultipleChoiceQuestion(LEGO_TOURIST_RULES, name, facts)
        # If facts were collected, proceed with ranking question; otherwise, default to a Loonie hypothesis
        if facts:
            facts, hypothesis = askRankingQuestion(LEGO_TOURIST_RULES, name, facts)
        else:
            hypothesis = "(?x) is a Loonie"  # Default hypothesis if no facts were gathered
        # Ask yes/no questions based on the gathered facts and hypothesis to refine the tourist identification
        result = askYesNoQuestion(LEGO_TOURIST_RULES, name, facts, hypothesis)

    print("\n============================")
    print("Thank you for participating!")
    print("============================\n")

    return result

def askMultipleChoiceQuestion(rules, name, result):
    """
    Asks a multiple-choice question to gather information about which activity the person (identified by 'name')
    engages in. Based on the user's selection, the function updates the facts (result) and applies forward 
    chaining to infer additional information.

    Args:
        rules (list): A list of rules in the expert system.
        name (str): The name of the person being evaluated.
        result (tuple): Current set of facts (answers provided so far).

    Returns:
        tuple: Updated facts after processing the user's selection.
    """
    
    # Display a multiple-choice question to the user
    print(f"\nWhich of the following activities does {name} engage in?")
    # List of available choices representing different activities
    choices = [
        "1. Attend themed events and conventions",
        "2. Purchase LEGO merchandise",
        "3. Engage in building and collecting LEGO sets",
        "4. Take photos of themed LEGO builds",
        "5. None of the above"
    ]
    # Print each choice for the user to select
    for choice in choices:
        print("\t" + choice)

    # Get and validate the user's response
    while True:
        try:
            # Ask user to input a number corresponding to their choice
            response = int(input(f"Choose the number corresponding to the action {name} performs (1-5): "))
            if response < 1 or response > 5:
                raise ValueError  # Raise an error if input is out of range
            break
        except ValueError:
            print("Invalid choice. Please enter a number between 1 and 5.")  # Handle invalid input

    # Assign a fact based on the user's selection
    if response == 1:
        fact = "(?x) attends themed events and conventions"
    elif response == 2:
        fact = "(?x) purchases merchandise"
    elif response == 3:
        fact = "(?x) engages with building and collecting activities"
    elif response == 4:
        fact = "(?x) takes photos of themed builds"
    else:
        fact = None  # If "None of the above" is selected, no fact is assigned
    # If a valid fact is generated, update the result with the new fact
    if fact:
        y = list(result)
        y.append(fact.replace("(?x)", name))
        result = tuple(y)
    # Apply forward chaining to update the result with any inferred facts
    result = forward_chain(rules, result)
    # Display the user's selected choice
    print(f"\nYou selected: {choices[response - 1].split('. ', 1)[1]}")

    return result

def askRankingQuestion(rules, name, result):
    """
    Asks the user to rank a set of activities in terms of how often the person (identified by 'name')
    participates in them, using a 1-5 ranking system. Each ranking corresponds to a potential tourist type,
    and the function derives facts and hypotheses based on the user's input.

    Args:
        rules (list): A list of rules in the expert system.
        name (str): The name of the person being evaluated.
        result (tuple): Current set of facts (answers provided so far).

    Returns:
        tuple: Updated facts and a list of hypotheses based on the ranking input.
    """
    
    # Prompt user to rank the activities from 1 to 5
    print(f"\nRank the following activities in order of {name}'s attendance (5 = Most attended, 1 = Least attended):")
    # List of activities corresponding to different tourist types
    preferences = [
        "Engaging with mechanics and engineering models",
        "Participating in modular building activities",
        "Participating in Star Wars-related events and screenings",
        "Engaging with urban planning activities",
        "Participating in adventure-based activities"
    ]
    # Display the activities for the user to rank
    for i, preference in enumerate(preferences):
        print(f"\t{i + 1}. {preference}")
    rankings = {}  # Dictionary to store user's ranking
    hypothesis = []  # List to store hypotheses based on the rankings

    # Loop through each activity and get a ranking from the user
    for preference in preferences:
        while True:
            try:
                # Ask user to enter a rank between 1 and 5 for the current preference
                rank = int(input(f"Enter your rank for {preference} (1-5): "))
                if rank < 1 or rank > 5:
                    raise ValueError("Rank should be between 1 and 5.")
                if rank in rankings.values():
                    raise ValueError("Each rank must be unique.")  # Ensure no duplicate ranks
                rankings[preference] = rank  # Store the rank for this activity
                break
            except ValueError as e:
                print(e)  # Handle invalid input and prompt again

    # Sort the activities by their rankings (highest rank first)
    sorted_rankings = sorted(rankings.items(), key=lambda item: item[1], reverse=True)
    fact = []  # List to store facts derived from rankings

    # Loop through sorted preferences and assign facts and hypotheses
    for preference, rank in sorted_rankings:
        # Assign facts and hypotheses based on the ranked preferences
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
    # Add the derived facts to the result
    if fact:
        for element in fact:
            y = list(result)
            y.append(element.replace("(?x)", name))
            result = tuple(y)
    # Apply forward chaining to update the result with any inferred facts
    result = forward_chain(rules, result)
    # Display the user's rankings
    print("\nYour ranking:")
    for preference, rank in sorted_rankings:
        print(f"\t{preference}")

    return result, hypothesis  # Return the updated result and the list of hypotheses

def askYesNoQuestion(rules, name, result, hypothesis):
    """
    Asks yes/no questions to gather additional facts based on the rules and current hypothesis.
    
    This function processes both single and multiple hypotheses, asking the user yes/no questions
    to confirm or deny conditions. It then updates the facts (result) based on the answers 
    and applies forward chaining to derive additional conclusions.

    Args:
        rules (list): A list of rules in the expert system.
        name (str): The name of the person being evaluated.
        result (tuple): Current set of facts (answers provided so far).
        hypothesis (str or list): The hypothesis (or list of hypotheses) being evaluated.

    Returns:
        tuple: Updated facts after processing user responses.
    """
    found = False  # Tracks if a conclusion is found based on the user's responses

    # Check if the hypothesis is a list (for multiple hypotheses)
    if isinstance(hypothesis, list):
        # Iterate through each specific hypothesis
        for specific_hypothesis in hypothesis:
            for rule in rules:
                # Check if the rule's consequent matches the hypothesis
                for statement in rule.consequent():
                    if statement == specific_hypothesis:
                        # Ask yes/no questions for each condition in the rule's antecedent
                        for condition in rule.antecedent():
                            # Skip conditions that are already in the result (already confirmed)
                            if condition.replace("(?x)", name) not in result:
                                print(condition.replace("(?x)", name) + "?")
                                while True:
                                    response = input("Enter 'yes' or 'no': ").lower()
                                    if response == "yes":
                                        # Update the result with the confirmed condition
                                        y = list(result)
                                        y.append(condition.replace("(?x)", name))
                                        result = tuple(y)
                                        # Apply forward chaining to derive additional facts
                                        result = forward_chain(rules, result)
                                        break
                                    elif response == "no":
                                        break
                                    else:
                                        print("Invalid input. Please enter 'yes' or 'no'.")
                                        continue
                                
                                # Check if a tourist type has been identified
                                for element in result:
                                    formatted = element.split(" ", 1)
                                    if len(formatted) > 1:
                                        element = f"(?x) {formatted[1]}"
                                    if element in touristTypes:
                                        print("\n " + "--- " + element.replace("(?x)", name) + " ---")
                                        found = True
                                        break
                            else:
                                continue  # Skip already known conditions
                if found:
                    break  # Stop further processing if a tourist type is found

    # Handle the case where the hypothesis is a single statement
    else:
        for rule in rules:
            for statement in rule.consequent():
                if statement == hypothesis:
                    # Ask yes/no questions for each condition in the rule's antecedent
                    for condition in rule.antecedent():
                        if condition.replace("(?x)", name) not in result:
                            print(condition.replace("(?x)", name) + "?")
                            while True:
                                response = input("Enter 'yes' or 'no': ").lower()
                                if response == "yes":
                                    # Update result with the confirmed condition
                                    y = list(result)
                                    y.append(condition.replace("(?x)", name))
                                    result = tuple(y)
                                    # Apply forward chaining to derive additional facts
                                    result = forward_chain(rules, result)
                                    break
                                elif response == "no":
                                    break
                                else:
                                    print("Invalid input. Please enter 'yes' or 'no'.")
                                    continue
                            
                            # Check if a tourist type has been identified
                            for element in result:
                                formatted = element.split(" ", 1)
                                if len(formatted) > 1:
                                    element = f"(?x) {formatted[1]}"
                                if element in touristTypes:
                                    print("\n " + "--- " + element.replace("(?x)", name) + " ---")
                                    found = True
                                    break
                        else:
                            continue  # Skip already known conditions
            if found:
                break  # Stop further processing if a tourist type is found

    # If no conclusion is reached, inform the user
    if not found:
        print(f"\n{name} is a Loonie.")
    
    return result 
