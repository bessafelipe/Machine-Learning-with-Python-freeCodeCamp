# This entrypoint file to be used in development. Start by reading README.md


from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player


#play(player, quincy, 1000,verbose=0)
play(player, abbey, 200,verbose=0)

play(player, kris, 1000,verbose=0)
play(player, mrugesh,1000,verbose=0)

# Uncomment line below to play interactively against a bot:
#play(human, abbey, 150, verbose=True)

# Uncomment line below to play against a bot that plays randomly:
# play(human, random_player, 1000)



# Uncomment line below to run unit tests automatically
# main(module='test_module', exit=False)