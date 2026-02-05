# CRBot
A clash royale bot that plays the game for you. Currently developing

## Elixir Detection Logic (elixir.py) 
The bot needs to know how much elixir the player currently has so it does not attempt to play cards without enough resources. To achieve this, the elixir bar is analyzed directly from the screen using pixel color detection.

First, a small rectangular region is captured as a screenshot that contains only the elixir bar. This region is adjusted for my screen and can be modified if needed. Then, purple pixels are detected in the screenshot. As the filled part of the elixir bar is purple, a target purple color is defined to detect the pixels: `PURPLE_COLOR = [208, 33, 214]` with a tolerance value `TOLERANCE = 80`. This tolerance value allows some difference between the pixel colors and the target color to make detection still work if there are minor color variations. Each pixel in the screenshot is compared to this target purple color, for example:
```
Pixel: [200, 50, 220]
Target: [208, 33, 214]

Difference:
|200-208| = 8
|50-33|  = 17
|220-214|= 6

So:
diff = [8, 17, 6]
```
This compares every pixel to the target purple. After this, each pixel is checked if it's within tolerance in R, G, and B channels. This creates a True/False mask:
```
For each pixel, if:
Red difference ≤ tolerance
AND
Green difference ≤ tolerance
AND
Blue difference ≤ tolerance
Then:
True (purple pixel)
Otherwise:
False
```
So the result is a boolean mask: `[True, True, False, True, ...]` The True values are counted which is the sum of purple pixels. This gives the total number of purple pixels in the elixir bar. Based on testing, the number of purple pixels was measured for each elixir level from 0 to 10, and from those measurements threshold values were created. For example, if purple pixels < 223 (the first threshold value) -> elixir=0. If between 223 and 544 (first and second threshold values) -> elixir=1, and so on. This converts the pixel count into an integer elixir value from 0 to 10.

There is one special case that needs to be handled. When the elixir reaches ten, the game briefly shows a pink "Elixir bar is full" message over the bar which messes up the purple pixel count. During this moment the number of purple pixels drops and pink pixels increase. To prevent this the code also checks for the pink color associated with the full elixir message. If enough pink pixels are detected, the system immediately returns an elixir value of ten regardless of the purple pixel count. 

## NOTES TO ADD LATER
Add what users might need to change in order to get the bot working for their own computer:
1) Elixir pixel colors and threshold -> If you are playing the game with different colors such as inverted colors, darkened, or any other color, the elixir count might malfunction. The current elixir number is calculated by counting pixels of the elixir bar, if the colors are different from the normal purple color, the counting might not work properly.
2) Screenshot region of elixir bar
3) Game region