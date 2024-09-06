from production import forward_chain
from rules_example_zookeeper import ZOOKEEPER_RULES, ZOO_DATA
from rules import LEGO_TOURIST_RULES, LEGO_TOURIST_DATA


if __name__=='__main__':
    result = forward_chain(LEGO_TOURIST_RULES, LEGO_TOURIST_DATA)
    print(result)