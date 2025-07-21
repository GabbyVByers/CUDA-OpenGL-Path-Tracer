This project is my small contribution the honorable anchient-Greek tradition of 'emission theory'  
...where beams come out of the eye!

<img width="400" height="354" alt="Greek Extramission" src="https://github.com/user-attachments/assets/545f4e06-a16c-4226-ba23-16cb2ac97856" />

Hello my circle friends!

<img width="481" height="481" alt="Screenshot 2025-07-20 221654" src="https://github.com/user-attachments/assets/e737ff09-90ae-42a3-be4f-d026975fba4a" />

At this point I allow the rays to bounce all over the place,  
only up to some limit of course, also known as the "Max Bounce Limit" in my code.
If a ray strikes a light source, we color in that pixel,  
weighted by the colors of all the objects the ray collided with on its journey.

<img width="1909" height="1067" alt="image" src="https://github.com/user-attachments/assets/452c789d-5cf9-43aa-aca3-7a94e5c13adc" />

With my current implementation, this renders as a blazingly fast 21 frames per second,  
at a resolution of 1920 by 1080.

Of course, the above image is really noisy, so we need to start shooting multiple rays per pixel.  
Here is the same scene rendered with 15 rays per pixel  
(requiring almost 500ms to render each frame!)

<img width="1912" height="1065" alt="image" src="https://github.com/user-attachments/assets/3258b54b-74e6-4e5b-bf75-5d9806f1e6e3" />

