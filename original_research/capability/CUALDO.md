# Computer Use Action Language Duo Output  

a robotics attempt that adds to an existing visual language model. train a FFW + a linear that starts at the end of the transformer BEFORE the decoding to tokens. after the FFW or several FFWs it decodes directly to action space. namely for each motor it has a torque and a duration. this can be trained with RL and physics simulations. . 

CURRENTLY! obviously don't have robots for now. but we CAN do computer use also with actions. these are the computer actions (copied from INT3RF4CE repo):  

mouse commands:

- MOVE x y (move cursor to position)
- LDOWN (push down left mouse) 
- LUP (release left mouse)
- RDOWN
- RUP
- SCROLLLOW
- SCROLLHIGH

keyboard commands:

- TYPE (special one! WHILE THIS IS ACTIVE, whatever the model "predicts" as thinking tokens is TYPED to the keyboard!)
- SPECIAL (this BUFFERS special keys to be released at once. make sure the model has all special lower case keys such as "enter" "shift", "alt", "tab", "ctrl" IN TOKENIZER! i.e. enter while SPECIAL is active will for example execute something in terminal but enter while TYPE is active will literally type the word "enter". i.e. while SPECIAL is on if the model thought ctrl, then shift, then s, they will be launched to the computer at once)

IF multiple actions are used at the same time we do them in the order from top to bot as they arementioned above.


so at the last layer we add an aditional linear projection to the weigths of size (embed_dim, 11) to match the commands and arguments. some are tokenized as bools. so the linear should have both weights and bias (negative bias to ignore noise) and then a relu. so anything > 0 would be treated as True and 0 would be treated as False. some are made input ints such as x (0-1920) and y (0-1080)

we feed the model the new screenshot as soon as it inferences the new token. it should learn to ignore the same inputs if it is unuseful. there are no text inputs. pure image.  

WE HOOK THE MODEL WITH INT3RF4CE TO ACTUALLY MAKE IT INTERACT WITH THE COMPUTER. DO NOT RECODE A WRAPPER!  

## the curriculum

we build the enviroment for it to learn.

### clicking  

maybe we just randomly spawn circles for it to click? with a prompt guiding it cause it is assumed it understands language already. different color means different click? left or right. and maybe later we add like drag circle to square and scroll to like find more circles?  
it's kinda like fps aim training.  

### typing  

we train it to type maybe by just prompting it stuff that if it types correctly out it gets a reward? and then prompting it to try comboing?  

### combined  

prompt to copy word. click bubble with a sentence on top, then type it in, then speical enter. correct to gain points.  

### reasoning 
prompt to answer questions. click bubble for simple math questions on top. type the correct verifiable answer. then special enter. correct to gain points.  


### app using

calculator
terminal

### complex app using

browser
code editor

### optional games benchmarks
(this will be fun. maybe not that useful. we literally watch it play games)  

Minesweeper
Tetris
MS Paint (creative)
OSU



## some thoughts

we dont train a model from scratch. we use an existing VLM into a VLA that can use computers with minimal additional weights.  
eventually merge this enviroment into a folder of D4TA5TEPS. maybe just this repo as a "action" folder.  

we use Qwen2.5-VL-3b for now

this is the most direct easiest way to train a sense of embodiment into LLMs. and it might have more positive effects than smooth computer use. it might help produce a sense of agency in the model. seeing the effects of it's actions. exploring an enviroment. et cetra.  
also interesting that we can add arbitiary decoding heads to a model. maybe this 11 dimensions goes towards using computers and another several dimensions goes towards the torque of a robot. just need to train it.  

