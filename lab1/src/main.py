from production import forward_chain, backward_chain
from rules_example_zookeeper import ZOOKEEPER_RULES, ZOO_DATA
from rules import LEGO_TOURIST_RULES, LEGO_TOURIST_DATA


if __name__=='__main__':
    result = backward_chain(ZOOKEEPER_RULES, 'dinu is a zebra')
    print(result)