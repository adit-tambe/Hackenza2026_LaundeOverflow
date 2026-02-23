# U-Flash: Democratizing Underwater Optical Communication
**A Hackathon Project Proposal**

**Team Name:** Launde Overflow
**Team Members:** Adit, Samyagya, Chitraksh, Vinit

---

## 1. Executive Summary

Communicating underwater has always been a tough problem. It's usually restricted to expensive, highly specialized equipment. Acoustic systems (using sound) can reach far, but they are incredibly slow and suffer from echoes in shallow waters. On the other hand, traditional optical systems (using light) offer fast speeds but require custom laser equipment, precise alignment, and thousands of dollars.

**U-Flash** is our software-based solution to this problem. We transform standard, everyday smartphones into reliable underwater communication devices. Instead of ignoring or fighting the way water scatters light, we actually use that scattering to our advantage. By combining this insight with advanced error-correction software running right on the phone, we've built an affordable way for divers to text each other, sensors to send data, and underwater vehicles to communicate.

Our team, Launde Overflow, believes in making technology accessible. Rather than building a custom piece of hardware, we wrote complex software to overcome the limits of regular consumer phones (specifically, their standard 30 fps cameras and LED flashlights). This proposal walks through our journey from defining the problem to testing our final prototype.

---

## 2. The Problem: Why is Underwater Communication so Hard?

The underwater environment is incredibly hostile to the digital signals we use every day.

- **Radio Frequency (RF) signals** (like WiFi, Cellular, and Bluetooth) die out within a few centimeters because water is highly conductive.
- **Acoustic waves** (sound) are the current industry standard, but they travel slowly. Echoes bouncing off the surface and the sea floor create confusing noise (multi-path fading). Plus, they just can't carry enough data for modern needs—often maxing out at a few kilobits per second.
- **Optical Wireless (Light)** offers a great alternative because blue-green light can travel decently well through water. However, the light gets scattered by water molecules and any dirt or particles floating around.

The core problem we tackled for this hackathon was **accessibility**: *How can we create a reliable underwater communication link without asking users to buy expensive, custom-built modems?*

We decided to focus entirely on engineering a practical, software-driven system using hardware everyone already owns: a smartphone’s LED flashlight as the transmitter, and its camera as the receiver.

---

## 3. Ideation: How We Got Here

During the brainstorming phase, Team Launde Overflow evaluated several ideas. We forced ourselves to reject anything that strayed from our "zero-custom-hardware" philosophy.

### 3.1 Ideas We Considered & Rejected

**Idea 1: Sending Data via Sound (Speaker to Microphone)**
- *The Concept:* Modulate data into high-frequency audio tones, play it through one waterproof phone's speaker, and listen with another phone's microphone.
- *Why we rejected it:* We tested this early on, and the results were very poor. In shallow water, the sound echoes bounced around too much, making the signal impossible to decode beyond a meter. Additionally, water physically presses against a phone speaker, severely distorting the sound it tries to make.

**Idea 2: Custom LED and Sensor Hardware**
- *The Concept:* Build an external rig using powerful blue LEDs and highly sensitive light detectors, connected to the phone via Bluetooth.
- *Why we rejected it:* While this would work great for range and speed, it completely violated our main goal: accessibility. We wanted a diver to be able to just install an app on their phone, not buy and carry a custom $150 hardware rig.

**Idea 3: Simple Flashing (On-Off Keying)**
- *The Concept:* Send data by flashing the phone's LED on and off, and have the receiving camera read the overall brightness of the video frame.
- *Why we rejected it:* It's intuitive, but standard phone cameras only shoot video at 30 frames per second (fps). That means we could only send about 15 bits of data per second. A simple text message would take minutes to arrive.

### 3.2 Our Accepted Breakthrough: The Rolling-Shutter Trick

To send data faster than 30 frames per second, we had to rethink how we used the camera. We took advantage of the **rolling-shutter mechanism**.

Instead of capturing the whole image at once, a typical phone camera reads pixels row-by-row from top to bottom. If you point it at a rapidly flickering LED, the camera doesn't see a solid light; it sees horizontal bright and dark "stripes" across a single frame. This clever trick means we sample the light thousands of times per second (as each row is read) rather than just 30 times a second.

**Using Scattering as an Asset:**
Usually, when light scatters underwater, it blurs the signal. If we were doing this in the air, trying to perfectly align two cameras so the light hits the exact right pixels would be a nightmare. But underwater, the scattering causes the light source to bloom and look larger. We realized this was actually helpful! The light hits a larger area of the camera sensor all at once, meaning we don't have to aim the phones perfectly at each other. As long as our software could handle the blur, the scattering was our friend.

---

## 4. How We Built It: The Architecture

We replaced custom hardware with smart signal processing. Our system is built in layers, much like how the internet works, taking raw text, safely pushing it through the challenging ocean environment, and decoding it on the other side.

### 4.1 Transmitting the Light (Physical Layer)

Instead of just blinking the light on and off—which gets easily confused by sunlight patches moving underwater—we used **4-Pulse Position Modulation (4-PPM)**.

- **How it works:** We group time into small windows of 4 slots. Based on the data, we flash the light in exactly one of those 4 slots. The receiver just has to find which of the 4 slots was the brightest. This is much more reliable because it doesn't matter if the water suddenly gets a bit brighter or darker; we are only comparing the 4 slots against each other.
- **Keeping the Camera Awake:** Phone cameras auto-adjust their exposure. If we send too many '0's and leave the LED off for too long, the receiving camera gets confused. We added special coding (Run-Length Limited) to ensure the light always flashes at least once every few milliseconds, keeping everything synchronized.

### 4.2 Error Correction: Our Three-Tiered Defense

The underwater channel is chaotic. A single bubble crossing the beam can block the light long enough to destroy hundreds of data bits (a "burst error"). Simple error checking wouldn't cut it, so we built a robust, three-tiered defense pipeline:

1. **Viterbi Decoding (The First Pass):** We used an algorithm commonly used in deep-space communication. Instead of making hard "yes, that’s a 1" or "no, that’s a 0" guesses, our software calculates probabilities based on the exact color blending in the image. This means we can salvage partial data even if the image is muddy.
2. **Block Interleaving (Fighting Bubbles):** To protect against a bubble wiping out a whole chunk of a word, we scramble the bits before sending them. We literally shuffle the data across different frames. If a bubble blocks the light, it only takes out scattered pieces of the overall message, which is much easier to fix later, rather than destroying an entire word at once.
3. **Reed-Solomon (The Final Polish):** As our ultimate safety net, we apply Reed-Solomon coding. This relies on advanced math to seamlessly repair any remaining scattered errors that slipped past our first two defenses, delivering clean, perfect text to the user.

### 4.3 The "Smart" Features

We wanted to leverage the fact that smartphones are powerful computers with many sensors:

- **Motion Compensation (Software Stabilization):** Divers hold phones in shaky hands. When the phone shakes, the light shifts on the camera. We tap into the phone's gyroscope to detect exactly when it shakes. When the software feels a tremor, it tells the decoder to place less trust in the light received during that exact split-second, relying instead on our error-correction safety net. This simple trick slashed our error rate dramatically.
- **Sensing Water Quality:** Our system can tell how muddy the water is. By analyzing how much the light scatters (how big the glow is on the camera), we can estimate the water's turbidity. So, while sending a message, the phone also acts as an environmental sensor.
- **Predicting the Future with AI (LSTM):** We added a small neural network into the system. It monitors how clear the signal has been over the last 20 seconds and predicts how bad it will be 5 seconds from now. If it predicts the diver is entering very murky water, the app automatically slows down the data rate and increases error correction to ensure the connection isn't lost.

### 4.4 The User Interface

We packaged all this math into a beautiful, mobile-friendly web application.

- **The Look:** We went with a "Dark Ocean" theme that looks sleek and modern, avoiding heavy, battery-draining frameworks. 
- **Real-Time Data:** Using JavaScript, we built a dashboard that shows live graphs of the signal waves and error rates as they happen.
- **The Brains:** A Python server runs in the background. It handles all the intense math and seamlessly passes the final decoded messages to the front-end dashboard without lagging the user's phone.

---

## 5. How We Worked: Development Phases

Building this was a big challenge, so Team Launde Overflow broke it down into structured phases.

**Phase 1: Research and Math (Weeks 1-2)**
- We dove deep into optical physics, studying how light behaves in seawater.
- We built basic Python models to simulate how a rolling-shutter camera sees flashing LEDs underwater. We used these models to confidently reject our initial, flawed acoustic ideas.

**Phase 2: Building the Core (Weeks 3-5)**
- We coded the heavy math first: the Viterbi decoder, the data scrambler (interleaver), and the Reed-Solomon error correctors. We chose to write these from scratch so we understood every part, paving the way for putting it on an actual phone later.

**Phase 3: Connecting the Pieces (Weeks 6-7)**
- We built the `SimulationEngine`. This piece of software tied everything together—taking text, scrambling it, simulating the hostile underwater environment, and then carefully unwrapping the data on the other side.
- We also integrated our AI prediction model and the gyroscope motion-compensation logic.

**Phase 4: The Interface (Weeks 8-10)**
- We converted our terminal-based data into an interactive web app.
- We designed the 5 main screens: the Home Dashboard, the Transmit screen (with easy-to-use sliders for water conditions), the Results viewer, Settings, and Documentation.

**Phase 5: Fixing Bugs and Polishing (Weeks 11-12)**
- We hit a huge roadblock: our first version of the Viterbi math was incredibly slow, freezing the app. We had to rewrite how we processed arrays, speeding it up by 90%.
- We ran heavy automated stress tests to ensure our error correction could survive massive simulated bubble blockages.

---

## 6. Challenges (and What We're Still Working On)

Getting consumer hardware to perform like industrial equipment wasn't easy.

1. **The Math was Too Slow:** As mentioned, our initial code was too slow for a phone processor. We spent days re-writing loops into highly optimized array calculations to get the decoding speed under a second.
2. **Finding the Right Blend for Bubble Errors:** Tuning our interleaver (the data scrambler) was tough. We had to find the exact mathematical balance between scrambling the data enough to survive a big air bubble passing by, without scrambling it so much that the process took ten seconds to decode.
3. **Keeping the UI Smooth:** Sending thousands of data points per second from the backend to draw a live graph crashed the mobile browser. We solved this by writing a "down-sampling" algorithm. It only sends the visual peaks and valleys of the graph to the user's screen, keeping the app snappy and the graph accurate.

**What we are still fighting:**
- **Sunlight Glare (Caustics):** When sunlight dances through shallow water, it creates bright, moving reflections on the camera. Our software occasionally confuses these bright flashes for our LED signal. We are currently working on better filters to distinguish sunlight from our data pulses.

---

## 7. What's Next? (Future Scope)

The software we’ve built is just the foundation. We see a massive future for this approach:

1. **A True Android App:** Our next step is to port our Python math directly into C++ and hook it into the Android camera system. This will turn our prototype into a fully deployable app that divers can take into the ocean tomorrow.
2. **Augmented Reality (AR) Displays:** Imagine a diver looking at their phone screen underwater and seeing incoming text messages and water quality stats floating in 3D space. We plan to integrate this with AR technology to create a cheap, effective heads-up display.
3. **Connected Robot Drones (AUVs):** Researchers often use small underwater drones. Instead of buying expensive acoustic modems for each drone, they could use our software and simple LEDs to let the drones talk to each other and coordinate searches in a mesh network.
4. **The Best of Both Worlds:** We want to build a hybrid system. Phones could use quiet, low-frequency sound chirps just to find each other from far away. Once the divers are facing each other, the app seamlessly switches to our high-speed optical flashing for actual chatting.

---

## 8. Conclusion

Team **Launde Overflow** has proven that the biggest hurdles in underwater digital communication aren't just hardware problems—they are software puzzles waiting to be solved. By accepting the harsh physics of water, using scattering to our advantage, and applying clever error correction from the aerospace industry, **U-Flash** offers a real, accessible bridge to the deep.

We believe that making technology affordable and accessible drives the best innovation. By removing the need for expensive, proprietary gear, U-Flash sets the stage for a new wave of underwater connection and exploration.

---
*Created for the 2026 Hackathon Prototype Phase.*
*Team Launde Overflow: Adit, Samyagya, Chitraksh, Vinit.*
