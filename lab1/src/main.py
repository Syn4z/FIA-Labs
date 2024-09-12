from production import forward_chain, backward_chain
from rules_example_zookeeper import ZOOKEEPER_RULES, ZOO_DATA
from rules import LEGO_TOURIST_RULES, LEGO_TOURIST_DATA
from questions import mainMenu


if __name__=='__main__':
    # result, humanReadable = backward_chain(LEGO_TOURIST_RULES, 'dinu is a LEGO Adventurers tourist')
    # print(humanReadable)
    # data = ('dinu is a tourist', 'dinu engages with urban planning activities', 'dinu shows interest in realistic cityscape and civic buildings')
    # result = forward_chain(LEGO_TOURIST_RULES, data)
    # print(result)
    result = mainMenu()