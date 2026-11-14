export type Slide = {
  id: string
  chapter: string
  duration: string
  takeaway: string
  notes: string[]
  cue?: string
  visual?: 'architecture' | 'intersection' | 'perception' | 'signal-flow' | 'score' | 'training' | 'metrics' | 'roadmap'
  demo?: {
    label: string
    target: 'live' | 'learn'
    scenario?: 'balanced' | 'east-jam' | 'pedestrians' | 'camera-loss' | 'fixed'
    steps: string[]
  }
  sources?: { label: string; href: string }[]
  appendix?: boolean
  kicker?: string
  title: string
  body?: string
  bullets?: string[]
  metricRows?: { label: string; value: string; note?: string }[]
  tables?: {
    caption: string
    headers: string[]
    rows: string[][]
  }[]
  footer?: string
  variant?: 'hero' | 'content' | 'dark-metric' | 'warning'
}

export const SLIDES: Slide[] = [
  {
    "id": "title",
    "chapter": "The project",
    "duration": "0:30",
    "variant": "hero",
    "kicker": "Classroom demonstration · Raphael Girard",
    "title": "From visible traffic\nto explainable decisions.",
    "body": "A working intersection simulation, real-image recognition, and a lab for teaching light choices.",
    "takeaway": "Observe the scene. Explain the decision. Measure the evidence.",
    "visual": "intersection",
    "notes": [
      "Opening: “My project investigates how a system can turn visible traffic into a proposed light decision, and how we can check whether that decision deserves our trust.”",
      "The demonstration combines a simulated intersection responding to screenshots with a real-footage monitor and a learning lab. The monitor samples a reusable licensed signalized-intersection clip, displays exact-frame visual outputs and collects pictures for explicit human review.",
      "State the boundary once at the start: this is a local research and teaching application. It is not connected to a city camera or physical traffic signals.",
      "The main presentation is designed for roughly 20 minutes, including demonstrations. Appendix slides supply extra technical detail for questions. The notes are a reference, not a script that must be read word for word."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      },
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "what-works",
    "chapter": "The project",
    "duration": "0:40",
    "variant": "content",
    "kicker": "01 · Establish the starting point",
    "title": "What can we demonstrate today?",
    "bullets": [
      "Working: pixel-based simulated control, clearance phases and decision logs.",
      "Working: pretrained real-image detection and saved human teaching examples.",
      "Measured: 81.0% object precision, 48.2% object recall on a public benchmark.",
      "Still to establish: useful learned light choices and performance on new intersections."
    ],
    "takeaway": "A functioning demonstration and a validated street controller are different milestones.",
    "notes": [
      "These are evidence snapshots from the 14 November 2026 project update. Label counts, trained-model status and live detections may change as you use the app; read the current lab status during the demonstration.",
      "The 81.0% value is object-detection precision. It does not mean 81% of light decisions are correct. Recall is 48.2%, so many labeled objects were missed.",
      "At this snapshot the learned light-choice model has no human training labels and is untrained. The running simulated controller works because it uses explicit rules. Do not imply that its moving lights prove a trained neural policy works.",
      "A successful class presentation can show working software and honest experimental failures together: the important result is that each claim has a distinct test and visible evidence."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      },
      {
        "label": "Measured detector benchmark",
        "href": "/classroom/evidence/PRETRAINED_DETECTOR_EVALUATION.md"
      }
    ]
  },
  {
    "id": "architecture",
    "chapter": "The project",
    "duration": "0:50",
    "variant": "content",
    "kicker": "02 · Four separate jobs",
    "title": "One interface. Four different questions.",
    "bullets": [
      "Screenshot controller: which simulated movement should receive green?",
      "Object detector: where are people, vehicles and road objects?",
      "Learning lab: which action would a human choose for this image?",
      "Historical LSTM: how much traffic arrives in the next hour?"
    ],
    "takeaway": "A detection score, an action probability and an hourly forecast have different meanings.",
    "visual": "architecture",
    "notes": [
      "Trace each branch of the diagram separately. In Live intersection, canvas pixels become observed traffic counts and waits; a deterministic policy chooses a phase; a state machine applies it after clearance.",
      "For a real street image, EfficientDet-Lite2 produces boxes and classes. A separate image classifier estimates the publisher’s binary anomaly label. Neither output supplies the correct light action.",
      "The light-choice model can learn from actions explicitly saved by a person. Its recommendations are reviewed; they are not automatically driving the live simulation.",
      "The historical LSTM panel uses stored hourly motorway data and trained forecasting weights. It does not receive the current photograph or control these lights.",
      "If asked “Which algorithm is the AI?”, explain that this system deliberately combines learned perception, simple supervised learning, explicit control rules and a separate time-series model. There is no single interchangeable accuracy score for all of them."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      },
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "interface-tour",
    "chapter": "Live operation",
    "duration": "0:30",
    "variant": "content",
    "kicker": "03 · Read the interface",
    "title": "Follow the same evidence the controller sees.",
    "bullets": [
      "One camera view displays the captured traffic image and detection boxes.",
      "The phase banner shows the current signal state, even when the camera image is frozen.",
      "The controller panel separates the next preview from the last applied decision.",
      "The decision log records the action, observations and actual phase duration."
    ],
    "takeaway": "Explain an applied decision using its captured evidence, not a moving preview.",
    "visual": "intersection",
    "cue": "Point to the camera preview, last applied decision and decision log before changing traffic.",
    "notes": [
      "Begin with the prepared balanced scene. Let at least one fresh screenshot arrive before discussing counts. Explain north/south as the vertical road and east/west as the horizontal road in this view.",
      "The preview can change while the current green is still running. That does not mean the lamp should change immediately; the state machine waits for the appropriate boundary.",
      "When a decision takes effect, the log keeps its measured observations, selected action, duration and capture time. Use that entry to explain what actually happened.",
      "The car, truck and pedestrian sliders configure the next Spawn traffic action. Moving a slider alone does not instantly add or remove road users. Pause freezes the simulation clock; Reset starts a new episode."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "screenshot-perception",
    "chapter": "Live operation",
    "duration": "0:50",
    "variant": "content",
    "kicker": "04 · Pixels become observations",
    "title": "The live controller really reads an image.",
    "bullets": [
      "Capture the hidden, unmodified 800 × 800 source canvas every 250 ms.",
      "Use color and shape components to locate drawn road users.",
      "Map detections to calibrated approaches and lanes.",
      "Pass measured observations to the controller."
    ],
    "takeaway": "These are real screenshot measurements of a simulated scene.",
    "visual": "perception",
    "notes": [
      "The screenshot recognizer receives RGBA pixels, the capture clock and prior image detections. It does not receive simulator vehicle IDs, off-screen arrival demand or the historical forecast.",
      "This is classical computer vision calibrated to the app’s artwork: color ranges and connected image components identify bodies and heads; known image regions identify the roads and lanes.",
      "A hidden 800 × 800 canvas supplies the raw image. Detection boxes are added to the single displayed camera view after capture, so the recognizer does not read its own boxes back into the next image. The preview pairs each capture with its matching detections.",
      "This approach is useful here because the road geometry, scale and colors are controlled. It is not a pretrained detector for arbitrary street photographs. Changing the artwork, crop or scale can break that calibration.",
      "A good answer to “Is the data fictional?” is: the environment is simulated, but the controller’s input is measured from its rendered pixels. Real-image detection is demonstrated separately with a model trained on photographs."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "tracking-and-queues",
    "chapter": "Live operation",
    "duration": "0:45",
    "variant": "content",
    "kicker": "05 · An image alone has no clock",
    "title": "Tracking adds waiting and movement.",
    "bullets": [
      "Match detections between captured frames.",
      "Estimate stopped traffic and pedestrian waiting time.",
      "Count a car as 1 queue unit and a truck as 2.",
      "Check whether the intersection box and crossings are occupied."
    ],
    "takeaway": "Counts describe demand; repeated observations add timing and clearance evidence.",
    "visual": "perception",
    "notes": [
      "A single still image cannot prove how long an object has been waiting. The simulator’s screenshot tracker uses repeated detections to estimate movement and stationary time.",
      "The queue score includes approaching vehicles that have not crossed the stop line, not only completely stopped vehicles. Stopped traffic contributes an additional term.",
      "Truck weight two is a hand-chosen representation of larger traffic demand, not a physical passenger-car-equivalent value calibrated from a real junction.",
      "The controller separately tracks time during which a nonempty direction is waiting for service. Those counters depend on usable observations; hidden simulation timers are not substituted when the camera fails.",
      "Occlusion, merged components and partly visible vehicles can still produce incorrect observations. Unknown colored fragments in the intersection are treated conservatively for clearance. Tests verify designed behavior but do not prove perception is perfect."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "score-example",
    "chapter": "Live operation",
    "duration": "1:00",
    "variant": "dark-metric",
    "kicker": "06 · A decision you can calculate",
    "title": "Why does east / west receive green?",
    "body": "Score = 2 × queue units + stopped units + 0.3 × red-wait seconds.",
    "tables": [
      {
        "caption": "Starting values · no higher-priority request",
        "headers": [
          "Direction",
          "Queue",
          "Stopped",
          "Red wait",
          "Score"
        ],
        "rows": [
          [
            "North / south",
            "4",
            "3",
            "10 s",
            "14.0"
          ],
          [
            "East / west",
            "10",
            "6",
            "5 s",
            "27.5"
          ]
        ]
      }
    ],
    "takeaway": "At the starting values, east / west wins and receives 4 + 0.5 × 10 = 9 seconds.",
    "visual": "score",
    "notes": [
      "Work the arithmetic slowly: north/south is 2×4 + 3 + 0.3×10 = 14. East/west is 2×10 + 6 + 0.3×5 = 27.5. East/west has the larger score.",
      "The duration rule is 4 + 0.5 times the selected direction’s weighted queue, clamped between 4 and 16 simulated seconds. Ten queue units therefore request nine seconds of through green.",
      "These values are an illustrative worked example, not a recorded benchmark result or a promise that the live preview will display these exact counts.",
      "The score coefficients and duration limits are explicit engineering rules. They were not learned by the LSTM, and no experiment has established that they minimize traffic delay.",
      "The assumptions matter: no pedestrian priority, no nonempty direction overdue at 30 seconds, and no operator request. Those conditions can override the ordinary score. A score winner still waits for clearance before receiving a green.",
      "If both directions have equal scores, the stored alternating direction breaks the tie. If only one direction has traffic, that direction is selected."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "priority-order",
    "chapter": "Live operation",
    "duration": "0:45",
    "variant": "content",
    "kicker": "07 · Rules around the score",
    "title": "The busiest queue does not always go first.",
    "bullets": [
      "At a clear boundary, eligible pedestrian requests can receive exclusive walk.",
      "A nonempty direction waiting at least 30 s gets fairness priority.",
      "A queued operator request is considered next.",
      "Otherwise, compare traffic scores and choose a 4–16 s green."
    ],
    "takeaway": "The policy balances visible demand, pedestrian service and waiting time.",
    "notes": [
      "Pedestrian walk is considered at an all-red decision boundary. A manual crosswalk request qualifies; automatic walk can qualify after a 20-second pedestrian wait, or after at least six pedestrians have waited at least 12 seconds, subject to the post-walk cooldown.",
      "Next, an overdue nonempty axis receives priority; if both qualify, the longer wait wins. Thirty seconds is a trigger at a future decision boundary, not a guaranteed maximum waiting time.",
      "An operator request is queued rather than switching the lamp immediately. It remains pending if pedestrian service or an overdue direction must happen first.",
      "Finally the normal pressure score chooses the axis. This ordering makes decisions inspectable, but these thresholds are simulator design choices rather than validated municipal timings.",
      "If the demonstration serves a pedestrian or the other direction before the jam, read the reason in the applied decision log. That is an opportunity to explain priority ordering, not a reason to pretend the controller ignored its input."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "phase-machine",
    "chapter": "Live operation",
    "duration": "1:00",
    "variant": "content",
    "kicker": "08 · Choosing and applying are separate",
    "title": "A requested green must pass clearance.",
    "bullets": [
      "Active green ends through a 1.4 s yellow phase.",
      "All-red lasts at least 0.8 s before a conflicting movement.",
      "Two fresh clear observations are required to release traffic.",
      "Protected left turns and a 6 s exclusive walk have their own phases."
    ],
    "takeaway": "The phase machine governs when a chosen movement may actually begin.",
    "visual": "signal-flow",
    "notes": [
      "The high-level controller chooses an axis or walk. The phase machine executes a sequence of permitted states instead of directly changing every lamp in response to a score.",
      "An ordinary axis finishes its through green and yellow. If observed left-turn demand exists, protected left turns may follow once straight traffic has cleared; they have their own green and yellow before the all-red boundary.",
      "All-red clearance is at least 0.8 seconds. Occupied crossings or an occupied intersection can extend the wait. Camera mode also needs two consecutive usable clear captures.",
      "Exclusive walk is nominally six simulated seconds, allowing the released pedestrian cohort to cross. If an observed crossing remains occupied, the transition waits. Later arrivals wait for another walk rather than extending the same cohort indefinitely.",
      "The 1.4-second yellow and other numbers are accelerated demonstration settings. They are not real-road timing recommendations or a certified safety design.",
      "Explain the distinction using a simple sentence: “The decision says who should go next; the phase machine determines when that request can be applied.”"
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "demo-congestion",
    "chapter": "Live demonstrations",
    "duration": "1:20",
    "variant": "hero",
    "kicker": "09 · Demonstration A",
    "title": "Create a queue.\nExplain the next green.",
    "body": "Watch the observed queue, the preview, then the applied decision log.",
    "takeaway": "A visible change in traffic should produce a traceable change in control.",
    "visual": "intersection",
    "cue": "Use the prepared east-jam scenario, then narrate one complete decision boundary.",
    "demo": {
      "label": "Demonstrate east-side congestion",
      "target": "live",
      "scenario": "east-jam",
      "steps": [
        "Find the east queue and its camera detections.",
        "Read the NS and EW score preview without assuming it is already applied.",
        "Wait through the current phase and clearance.",
        "Read the new applied action, reason and duration in the decision log."
      ]
    },
    "notes": [
      "Start the prepared east-jam scenario. Point to the new visible queue, not an internal jam flag. Explain that the controller sees the same captured image displayed in the camera panel.",
      "Expected observation: east-side demand increases the east/west queue contribution. At the next eligible decision boundary, that tends to favor east/west unless a pedestrian request or overdue direction has priority.",
      "The exact displayed queue and selected duration can vary as vehicles move and image detections update. Explain the actual logged reason rather than memorizing a single expected number.",
      "Do not promise an immediate lamp switch. A delayed change while yellow, an occupied box or pedestrian service completes is expected behavior.",
      "Optional audience question: “Would you choose the same direction? What extra information would you want?” Distinguish a preference from evidence that it improves delay.",
      "Return to the slideshow after one applied decision. A moving scene is a demonstration of mechanics, not a controlled experiment proving an efficiency improvement."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "demo-pedestrians",
    "chapter": "Live demonstrations",
    "duration": "1:00",
    "variant": "hero",
    "kicker": "10 · Demonstration B",
    "title": "Request a crossing.\nWatch vehicles clear first.",
    "body": "Pedestrian service is a distinct phase, not another vehicle direction.",
    "takeaway": "A request enters the controller; clearance still governs its execution.",
    "visual": "signal-flow",
    "cue": "Use the pedestrian scenario, request a crossing, then watch the phase label and the crosswalk.",
    "demo": {
      "label": "Demonstrate pedestrian service",
      "target": "live",
      "scenario": "pedestrians",
      "steps": [
        "Show pedestrians waiting at the curb.",
        "Use a crosswalk request if one is not already queued.",
        "Observe the current green finish, yellow and clearance.",
        "Confirm exclusive walk begins, then traffic resumes after the crossing clears."
      ]
    },
    "notes": [
      "Explain that NS and EW crosswalk buttons submit requests; the implementation uses one exclusive pedestrian phase in which vehicles are red.",
      "Expected observation: the request is visible before walk begins. Any vehicles still in the intersection must clear before the pedestrian phase releases the waiting cohort.",
      "Watch the phase label rather than only the drawn signal color. After walk, the queued vehicle direction resumes only after the crossing is observed clear.",
      "If someone asks why the button did not switch immediately, connect this back to the separation between choosing an action and applying a phase.",
      "Automatic pedestrian service has waiting-time rules and an 18-second post-walk cooldown. A manual button can request a walk without waiting for the automatic threshold.",
      "The demonstration does not validate accessibility timing for a real crossing. The walking speed and phase lengths are simulator parameters."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "demo-camera-loss",
    "chapter": "Live demonstrations",
    "duration": "1:00",
    "variant": "hero",
    "kicker": "11 · Demonstration C",
    "title": "Remove the camera.\nThe system must stop guessing.",
    "body": "A missing or stale screenshot blocks new green and walk phases.",
    "takeaway": "Loss of evidence must be visible in the behavior and status.",
    "visual": "perception",
    "cue": "Wait for a vehicle green, then switch Camera off. Read the phase banner and controller for yellow and the red hold while the camera image stays frozen.",
    "demo": {
      "label": "Demonstrate camera loss and recovery",
      "target": "live",
      "scenario": "camera-loss",
      "steps": [
        "Leave Camera on and wait for a vehicle green.",
        "Turn Camera off; the image freezes. Read the phase banner and controller as green finishes through yellow and holds all-red.",
        "Turn Camera on again.",
        "Wait for fresh clear frames and all-red clearance before service resumes."
      ]
    },
    "notes": [
      "The prepared scenario begins at all-red with Camera on. Let fresh captures establish clearance and wait for a vehicle green. Then turn the camera off. Its image stays at the last capture, so use the current phase banner and controller panel to show the complete yellow transition before the red hold.",
      "A usable capture must be valid, no later than the current simulation time, and at most one simulation second old. Missing, blank, failed-calibration and stale captures do not become empty-road evidence.",
      "When input is lost during green, the system runs the full yellow interval and then holds all-red. It never falls back to hidden simulator objects to keep choosing lights.",
      "Recovery requires usable new images, the all-red interval and clear observations. An operator request cannot bypass this condition.",
      "Pause also freezes the simulation clock, so it is not a camera-loss test by itself. Keep the simulation running when demonstrating stale-frame behavior.",
      "This is a tested fail-hold behavior within the simulated environment. It is not proof that all sensor failures or physical-world hazards have been handled."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "demo-fixed",
    "chapter": "Live demonstrations",
    "duration": "0:40",
    "variant": "content",
    "kicker": "12 · A useful comparison",
    "title": "Compare adaptive rules with a fixed cycle.",
    "bullets": [
      "Adaptive mode chooses from measured demand and waiting.",
      "Fixed mode alternates six-second through greens.",
      "Both retain turn, pedestrian and clearance behavior.",
      "A fair efficiency study needs identical arrivals and repeated runs."
    ],
    "takeaway": "A visual comparison explains the policies; it does not establish which performs best.",
    "demo": {
      "label": "Inspect the fixed-cycle comparison",
      "target": "live",
      "scenario": "fixed",
      "steps": [
        "Confirm Automatic traffic controller is off.",
        "Observe alternating through directions and six-second through greens.",
        "Inspect a decision-log entry labeled fixed cycle.",
        "Compare with the adaptive rule; avoid claiming a measured delay reduction."
      ]
    },
    "notes": [
      "This comparison is optional if time is short. The fixed-cycle scenario is useful for showing what the adaptive component changes.",
      "Switching automatic control off removes the score-based through-phase choice and uses six-second alternating through greens. It does not disable clearance, protected turns or pedestrian service.",
      "Different random arrivals can make one short run appear better by chance. A real comparison should replay the same arrival demand across policies and report mean and tail waiting time, throughput, pedestrian delay and fairness across many runs.",
      "The current presentation demonstrates behavior and tests its intended mechanics. It does not contain an experiment proving that the adaptive policy minimizes delay or beats a conventional traffic-engineering controller.",
      "If the audience asks “What does efficient mean?”, state the metric first. Passing more vehicles while leaving pedestrians waiting indefinitely would not be an adequate definition."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "pretrained-detector",
    "chapter": "Real-image recognition",
    "duration": "0:50",
    "variant": "content",
    "kicker": "13 · Transfer learned visual features",
    "title": "Real photographs use a pretrained detector.",
    "bullets": [
      "EfficientDet-Lite2 was pretrained on COCO object labels.",
      "It locates eight road-related classes, including people and vehicles.",
      "Large images also use overlapping crops to preserve small details.",
      "Inference is local; the detector weights are currently frozen."
    ],
    "takeaway": "Transfer learning provides a practical starting point, not a guaranteed aerial-traffic solution.",
    "visual": "perception",
    "notes": [
      "EfficientDet-Lite2 is Google’s pretrained object detector, run locally through MediaPipe. The selected classes are person, bicycle, car, motorcycle, bus, truck, traffic light and stop sign.",
      "Pretraining is useful because a large dataset has already taught visual features and object localization. This project does not try to learn those detector weights from a small collection of local examples.",
      "For images whose longest side exceeds 960 pixels, inference adds overlapping 640-pixel crops, spaced 512 pixels apart, to a whole-image pass. Boxes are remapped into the original image and same-class duplicates are merged.",
      "This adds computation: the public benchmark averaged about 280 milliseconds per image locally, while one 1920×1080 training frame required nine passes and roughly 2.38 seconds. These measurements are not an end-to-end live-video frame rate.",
      "No local detector fine-tuning has been done. The 3,000 aerial images have binary anomaly labels, not the object boxes required for supervised detector training.",
      "If privacy is asked about: inference runs on this machine and the app does not upload these images to a model service. MediaPipe may report SDK usage metrics; its maintainer states input images are not sent. The guide records that distinction."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      },
      {
        "label": "Measured detector benchmark",
        "href": "/classroom/evidence/PRETRAINED_DETECTOR_EVALUATION.md"
      }
    ]
  },
  {
    "id": "detection-evidence",
    "chapter": "Real-image recognition",
    "duration": "0:55",
    "variant": "dark-metric",
    "kicker": "14 · Read both sides of recognition",
    "title": "81% precision is promising.\n48% recall shows the missing work.",
    "metricRows": [
      {
        "label": "Precision",
        "value": "80.98%",
        "note": "How many reported objects were correct?"
      },
      {
        "label": "Recall",
        "value": "48.18%",
        "note": "How many labeled objects were found?"
      },
      {
        "label": "F1",
        "value": "60.42%",
        "note": "Balances precision and recall"
      },
      {
        "label": "Evaluation",
        "value": "500 images",
        "note": "Vehicle-containing COCO val2017 subset"
      }
    ],
    "takeaway": "The 90% recognition target is not met; these are not light-decision scores.",
    "visual": "metrics",
    "notes": [
      "Precision asks: of the detections that counted in scoring, what proportion matched a labeled object? Recall asks: of the eligible labeled objects, what proportion did we detect?",
      "The measured counts were 1,933 true positives, 454 false positives and 2,079 false negatives. Ninety-three detections matching crowd regions were ignored under the stated evaluation rule.",
      "A detection counts as a match only with the correct class and at least 0.5 intersection-over-union with the label. The detector score cutoff was fixed at 0.3 before evaluation.",
      "The images were selected before inference from a public vehicle-containing COCO validation subset. This does not measure false alarms on every empty scene, does not represent an independent aerial-junction holdout, and may overlap the model publisher’s development evaluation.",
      "The project’s recognition target requires at least 90% precision and recall for every displayed road class, with minimum sample support. Raising a displayed confidence threshold is not the same as meeting that target.",
      "Use the phrase “measured public-benchmark precision” instead of “81% reliable.” The distinction is central to explaining the project professionally."
    ],
    "sources": [
      {
        "label": "Measured detector benchmark",
        "href": "/classroom/evidence/PRETRAINED_DETECTOR_EVALUATION.md"
      }
    ]
  },
  {
    "id": "aerial-data",
    "chapter": "Real-image recognition",
    "duration": "0:45",
    "variant": "content",
    "kicker": "15 · More examples, with boundaries",
    "title": "3,000 unique aerial images. Recordings kept apart.",
    "tables": [
      {
        "caption": "Expanded UIT-ADrone experiment",
        "headers": [
          "Partition",
          "Images",
          "Source recordings",
          "Purpose"
        ],
        "rows": [
          [
            "Training",
            "2,550",
            "35",
            "Fit the anomaly classifier"
          ],
          [
            "Calibration",
            "225",
            "8",
            "Fit probability calibration"
          ],
          [
            "Test",
            "225",
            "8",
            "Measure held-out behavior"
          ]
        ]
      }
    ],
    "takeaway": "Separate recordings reduce leakage; more nearby frames do not create new environments.",
    "notes": [
      "The original collection contained 1,500 images. We added 1,500 distinct eligible training frames while retaining all 450 original calibration and test examples unchanged.",
      "Every image has a unique checked file hash. Training, calibration and test use disjoint source recordings. The selection maintains at least 60 source frames between included images under the original sampling rules.",
      "The expanded training set contains 1,412 normal and 1,138 anomaly frames. Calibration and test each remain 180 normal and 45 anomaly frames, or 80/20. The entire expanded collection is therefore not an 80/20 split.",
      "This distinction explains why “add more examples” is not enough by itself: the additional images come from the same 35 training recordings, rather than new cities, camera heights, weather conditions or junction layouts.",
      "The publisher provides binary normal/anomaly labels. These do not specify vehicle boxes, jaywalking, red-light violations or correct north/south light actions.",
      "The roundabout gallery is consequently useful for inspecting visual outputs, but it is disabled for teaching signal choices. A roundabout cannot honestly supply a correct signal action merely because it contains vehicles."
    ],
    "sources": [
      {
        "label": "Dataset and split audit",
        "href": "/classroom/evidence/STREET_VISION_DATA_EXPANSION.md"
      },
      {
        "label": "Expanded anomaly experiment",
        "href": "/classroom/evidence/STREET_VISION_EXPANDED_EVALUATION.md"
      }
    ]
  },
  {
    "id": "anomaly-result",
    "chapter": "Real-image recognition",
    "duration": "0:45",
    "variant": "warning",
    "kicker": "16 · Historical baseline, still the default",
    "title": "80% accuracy can hide complete failure.",
    "bullets": [
      "The historical/default classifier labeled every test frame normal.",
      "It missed all 45 anomalous frames: anomaly recall = 0%.",
      "“Always normal” also scores 80% on this 80/20 test set.",
      "Adding 1,500 training images did not fix this failure."
    ],
    "takeaway": "Compare against a simple baseline and report the errors the task actually cares about.",
    "notes": [
      "This is the historical expanded baseline, which remains the active/default anomaly model because later candidates failed validation. It is separate from EfficientDet-Lite2 and the light-choice model.",
      "The classifier used frozen ImageNet MobileNetV3Small features, training-only standardization and balanced L2 logistic regression. A sigmoid probability calibrator was fitted on separate validation images; the classification cutoff stayed fixed at 0.5.",
      "Its test confusion matrix was 180 true normal predictions and 45 missed anomalies, with no positive predictions. Overall accuracy was 80%, balanced accuracy was 50%, and anomaly recall was zero.",
      "Rank-based average precision improved retrospectively from 0.3099 to 0.3752 after adding data, but the output probabilities stayed near 20% and the fixed-threshold decisions remained unusable.",
      "Calibration did not create information the visual features lacked. A model can emit plausible-looking probabilities yet fail to distinguish the examples that matter.",
      "The same historical test set has been inspected across experiments. These results must be described as retrospective. A future improved model needs a new untouched recording set for confirmation."
    ],
    "sources": [
      {
        "label": "Expanded anomaly experiment",
        "href": "/classroom/evidence/STREET_VISION_EXPANDED_EVALUATION.md"
      }
    ]
  },
  {
    "id": "anomaly-finetuning",
    "chapter": "Real-image recognition",
    "duration": "1:00",
    "variant": "warning",
    "kicker": "Follow-up experiment · Actual visual fine-tuning",
    "title": "Visual weights changed. Validation still failed.",
    "bullets": [
      "A separate 320px MobileNetV3Small updated its final visual blocks using publisher labels.",
      "Separate recording validation found 1 of 21 anomalies: 4.76% recall; required ≥40%.",
      "The candidate was rejected and remains inactive. Training completion is not validation."
    ],
    "metricRows": [
      {
        "label": "Retrospective anomaly precision",
        "value": "37.5%",
        "note": "6 true alerts / 16 alerts"
      },
      {
        "label": "Retrospective anomaly recall",
        "value": "13.3%",
        "note": "6 found / 45 anomalies"
      },
      {
        "label": "Retrospective false alerts",
        "value": "10 / 180",
        "note": "170 normal frames correctly left unflagged"
      }
    ],
    "footer": "Previously inspected test set: TP 6 · FN 39 · FP 10 · TN 170. No new reliability claim.",
    "takeaway": "Keep the failed candidate out of use; use its errors to design the next independent experiment.",
    "notes": [
      "This experiment genuinely changed visual convolution weights. It used a separate ImageNet MobileNetV3Small at 320×320, mean/max pooling and a regularized 64-unit head. The original 224px action encoder and pretrained object detector were unchanged.",
      "The early image stem stayed frozen and was cached. Three head warm-up epochs were followed by five low-learning-rate epochs updating the final two inverted residual blocks and final convolution. Batch-normalization statistics and weights stayed frozen. Original images and deterministic mild brightness/contrast views used the publisher labels; no labels were invented.",
      "Training gradients used 2,082 frames from 27 recordings. A different 468 frames from eight recordings chose the checkpoint. From the existing validation partition, 106 frames from four recordings calibrated scores and 119 from four other recordings selected the operating threshold. Whole recordings stayed separate.",
      "The threshold-validation subset determined the 0.25823751 threshold under a 20% false-alert ceiling. Activation required at least 40% recall, at least 30% precision, at most 20% false alerts and positive calibration slope. Validation recall was only 1/21, or 4.76%; precision was 50% and one of 98 normal frames caused a false alert. The gate failed before test scoring.",
      "On the 225-image historical test, the candidate found six anomalies, missed 39, raised ten false alerts and correctly left 170 normal frames unflagged. Precision was 37.5%, recall 13.33%, balanced accuracy 53.89% and overall accuracy 78.22%. The historical test had already been inspected, so these numbers are retrospective rather than fresh reliability evidence.",
      "The candidate remains inactive. The default baseline is also inadequate; keeping a rejected replacement inactive does not make that baseline reliable. No verified anomaly labels were supplied for the included FLUID signalized footage. Single-frame appearance cannot establish red-light running or jaywalking without motion, geometry and signal context.",
      "For a live observation example, use the top navigation Real traffic videos, choose the included FLUID clip and Analyze this frame. Show actual boxes, any missed vehicles, anomaly output and policy abstention. Return to Slideshow and select the next teaching slide; the video never controls any light."
    ],
    "sources": [
      {
        "label": "320px fine-tuning method, validation and test",
        "href": "/classroom/evidence/STREET_ANOMALY_FINETUNING.md"
      },
      {
        "label": "Included real signalized-footage provenance",
        "href": "/classroom/evidence/REAL_TRAFFIC_VIDEO_SOURCE.md"
      }
    ]
  },
  {
    "id": "review-demo",
    "chapter": "Teaching decisions",
    "duration": "1:20",
    "variant": "hero",
    "kicker": "17 · Demonstration D",
    "title": "Teach a choice from\na signalized scene.",
    "body": "Teach from a simulated screenshot or an exact saved frame from Real traffic videos. The two action models stay separate.",
    "takeaway": "A human action label is new supervision; an object box alone is not a light label.",
    "visual": "training",
    "cue": "Open Train & review and choose Create a signalized practice scene. The roundabout gallery is for recognition only.",
    "demo": {
      "label": "Open the decision review workspace",
      "target": "learn",
      "steps": [
        "Choose Create a signalized practice scene, or upload your own signalized intersection footage.",
        "In the live scene, press Teach a decision from this screenshot to open that exact image.",
        "Press Analyze this image, review the frozen scene, then choose and save your own preferred action.",
        "Show the saved-label count and remaining training requirements; return to this slide."
      ]
    },
    "notes": [
      "Use a prepared signalized simulation scene for a reliable classroom teaching demonstration. Open Live intersection, allow a fresh screenshot, then choose Teach a decision from this screenshot or select App screenshot in the lab.",
      "The supplied real street gallery contains roundabouts. It supports inspection of detections and anomaly labels, but cannot provide verified north/south or east/west signal choices. Say this explicitly if someone clicks a gallery image and finds teaching disabled.",
      "For real footage, choose Real traffic videos in the top navigation, select the included licensed FLUID sample, then Analyze this frame or Observe video. Capture this frame & review opens the exact saved frame in the teaching workspace. A chosen model score is not a human label; inspect visible evidence before making your own judgement.",
      "Choose North / south, East / west, Pedestrian walk or Hold all red, then save. Do not automatically treat missing information as a correct all-red label; skip ambiguous frames unless you can justify the intended action.",
      "Saving stores your preference for that exact image. It does not immediately retrain the model. A suggested action still needs explicit human confirmation and saving.",
      "Do not save invented labels merely to make a progress bar fill. For class, demonstrate one real human choice and show the requirements. Correcting object boxes to retrain recognition is not implemented; these controls teach action preferences.",
      "Street and simulation examples train separate models. Only an explicitly reviewed simulation choice can be applied to the simulator, through the existing clearance controller.",
      "Real saved frames now also accept an explicit anomaly review: No anomaly visible, Visible anomaly or Cannot judge, with an optional observation note. These reviews stay separate from light preferences; no choice is preselected, and saving does not train the anomaly model. Video packages include these human reviews for a future matching training workflow."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      }
    ]
  },
  {
    "id": "decision-features",
    "chapter": "Teaching decisions",
    "duration": "0:50",
    "variant": "content",
    "kicker": "18 · What the learner receives",
    "title": "Turn the image into a compact learning input.",
    "bullets": [
      "576 frozen image features describe overall scene appearance.",
      "72 values summarize eight detected object classes in a 3 × 3 grid.",
      "Your saved label supplies one of four preferred actions.",
      "A small regularized classifier learns patterns in those examples."
    ],
    "takeaway": "The street learner combines scene features, observed objects and explicit human choices.",
    "visual": "training",
    "notes": [
      "The 576 image features are produced by the frozen MobileNetV3Small scene encoder. They are learned visual measurements, not human-readable counts or hand-written traffic rules.",
      "For street images, the pretrained detector contributes counts for eight road classes in each of nine image regions: 8×9 = 72 additional values. Counts use log(1 + count) so very large counts do not dominate linearly.",
      "The resulting street input has 648 values before training-only standardization. Simulation teaching uses its own image-feature domain rather than pretending the real-image detector is calibrated to the drawings.",
      "Regularized logistic regression learns a weight pattern for each action. Regularization discourages large unstable coefficients; frozen feature extraction keeps the number of trainable parameters smaller than full neural-network fine-tuning.",
      "This is a practical baseline because the initial action dataset is small and the fitting process is easy to reproduce and inspect. We have not established that it is the best possible architecture.",
      "It learns agreement with recorded preferences. It has no reward function for traffic delay, no proof of optimality and no automatic access to motion history. If the input omits the signal state or previous waiting time, the learner cannot reliably reconstruct it from an ambiguous still."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      }
    ]
  },
  {
    "id": "training-process",
    "chapter": "Teaching decisions",
    "duration": "0:50",
    "variant": "content",
    "kicker": "19 · What pressing Train actually does",
    "title": "Fit, calibrate and evaluate on different groups.",
    "bullets": [
      "Start with ≥80 human labels, ≥12 recording groups and ≥8 labels per action.",
      "Fit the scaler and classifier on training groups only.",
      "Fit probability temperature on separate calibration groups.",
      "Evaluate remaining groups; preserve their assignments on retraining."
    ],
    "takeaway": "Nearby frames must stay together so evaluation cannot quietly reuse the same scene.",
    "visual": "training",
    "notes": [
      "The initial split assigns roughly half the recording groups to fitting, a quarter to calibration and a quarter to evaluation. Each partition also needs sufficient samples, groups and action classes; the interface explains shortfalls.",
      "Eighty labels and twelve groups are minimum conditions for a small experiment, not sufficient evidence of a reliable street policy. Do not manufacture independent groups by renaming frames from the same video.",
      "A video keeps a stable recording identity. Extracted stills should share a recording name. All screenshots from one simulation episode share a group; Reset begins a new episode, but many similar episodes still do not prove diversity of real streets.",
      "The scaler learns means and scales from fitting data only. The classifier then learns coefficients with a fixed regularization setting. A scalar temperature fitted on calibration groups adjusts the probability distribution without refitting the classifier on test labels.",
      "Image identity is preserved with a hash and frozen snapshot. Re-labeling the same image updates its existing label instead of counting a duplicate; correcting labels invalidates the previous model until retraining.",
      "Inspecting evaluation results repeatedly makes later comparisons retrospective. For a final claim, reserve genuinely new recordings that were not used to choose the model or threshold.",
      "At the initial snapshot the model is untrained because the project has no human action labels. Showing that honest state is preferable to fabricating a trained policy for the presentation."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      }
    ]
  },
  {
    "id": "confidence-and-reliability",
    "chapter": "Teaching decisions",
    "duration": "0:55",
    "variant": "warning",
    "kicker": "20 · Confidence is a claim to test",
    "title": "A 90% score is not 90% demonstrated reliability.",
    "bullets": [
      "Recommendations require probability ≥0.90 and similar training support.",
      "Otherwise the model abstains and asks for human review.",
      "Measure agreement and coverage on held-out recording groups.",
      "The evidence target requires the interval’s lower bound to reach 90%."
    ],
    "takeaway": "Show how often a model is right and how often it is willing to answer.",
    "notes": [
      "A model probability describes its internal estimate for an action. Empirical reliability asks whether those estimates agree with human labels on data kept separate from fitting and calibration.",
      "The recommendation gate uses a top-action probability of at least 0.90 and a nearest-training feature cosine similarity of at least 0.5. The similarity check is coarse; it does not guarantee detection of unfamiliar roads.",
      "Coverage is the proportion of evaluation frames on which the model actually makes a recommendation after both gates. A model that answers only one easy case cannot be called useful merely because that answer is correct.",
      "The project’s evidence flag requires at least 40 recommended test frames, at least three held-out groups, at least five examples of each action and an approximate 95% Wilson interval lower bound of at least 90% agreement.",
      "Nearby video frames are correlated, so a frame-based Wilson interval can be optimistic even with group separation. This is descriptive evidence, not a certified guarantee for new recording sites.",
      "Agreement with human preferences also differs from safe or optimal control. Human labels can be wrong or inconsistent. A stronger study needs independent review and traffic outcome measurements.",
      "The interface explicitly distinguishes the score cutoff from the evidence target. The current project does not claim that changing either number makes the model reliable."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      }
    ]
  },
  {
    "id": "verification",
    "chapter": "Evidence and next steps",
    "duration": "0:40",
    "variant": "content",
    "kicker": "21 · What has been checked",
    "title": "Test software behavior. Measure model behavior.",
    "bullets": [
      "Controller checks: changing traffic, priorities, clearance and camera recovery.",
      "Learning checks: saved labels, duplicate identity and isolated recording groups.",
      "Model checks: real-image detection metrics and anomaly failure analysis.",
      "Remaining experiment: matched-policy traffic outcomes on unseen conditions."
    ],
    "takeaway": "Passing software tests and achieving useful recognition are separate kinds of evidence.",
    "notes": [
      "Software tests verify properties such as a stale camera blocking new release, queued requests respecting phase transitions, and hidden scene changes not changing image-driven observations.",
      "Learning tests verify persistence, domain separation, immutable image identity, independent calibration and abstention behavior. They cannot create correct human training labels.",
      "The object benchmark and anomaly reports test empirical model outputs. Their scope and failure modes remain visible even when all software tests pass.",
      "Avoid quoting a permanent test count on the projected slide because the suite changes as the app evolves. Use the current development verification report if asked for the exact count.",
      "For an efficiency claim, replay matched demand across controllers and report multiple outcomes: mean wait, long waits, throughput, pedestrian delay, fairness and unresolved queues. Repeat across independent seeds and scenario types.",
      "For a street-recognition claim, use newly labeled signalized intersection recordings with camera-level separation. A public photographic benchmark is a useful diagnostic, not a substitute for the target deployment setting."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      },
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      },
      {
        "label": "Measured detector benchmark",
        "href": "/classroom/evidence/PRETRAINED_DETECTOR_EVALUATION.md"
      }
    ]
  },
  {
    "id": "roadmap",
    "chapter": "Evidence and next steps",
    "duration": "0:50",
    "variant": "content",
    "kicker": "22 · Make the next version stronger",
    "title": "A concrete path from demonstration to evidence.",
    "bullets": [
      "Collect diverse signalized footage with object boxes, tracks and reviewed actions.",
      "Add box correction and fine-tune recognition; reserve new cameras for testing.",
      "Add sequence context, signal state and outcome-based controller comparisons.",
      "Only then evaluate a supervised learned-control loop in simulation."
    ],
    "takeaway": "Better labels and independent validation matter more than a larger image count alone.",
    "visual": "roadmap",
    "notes": [
      "First define the intended scene and task: a fixed overhead camera at a signalized junction, the classes to recognize, the lane map, and the action or violation being evaluated. A single generic anomaly label cannot supervise all of those tasks.",
      "Object-recognition improvement needs verified bounding boxes, difficult negative examples, small or occluded road users, night and weather variation, and recording-level holdouts. The app currently lacks a box-correction interface.",
      "Motion-dependent violations require tracked sequences and visible signal state. A stationary car in one photograph does not prove a red-light violation; a person in a roadway does not establish illegal crossing without jurisdiction and timing context.",
      "Action learning needs reviewed preferences with a consistent labeling policy and documented ambiguity. Expert disagreement should be inspected rather than averaged away as if every label were certain.",
      "Evaluate candidate perception and policies on unseen recordings, then run learned recommendations through the existing phase constraints in simulation under supervision. Compare outcomes against simple baselines before claiming an advantage.",
      "Connecting to physical signals would be a separate engineering and validation project involving authorized infrastructure, fault handling and traffic-engineering review. That is beyond this classroom demonstration.",
      "The immediate presentable deliverable is already concrete: explainable live behavior, actual real-image detections, a review workflow, measured limits and a plan whose next experiment can succeed or fail visibly."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      },
      {
        "label": "Measured detector benchmark",
        "href": "/classroom/evidence/PRETRAINED_DETECTOR_EVALUATION.md"
      },
      {
        "label": "Expanded anomaly experiment",
        "href": "/classroom/evidence/STREET_VISION_EXPANDED_EVALUATION.md"
      }
    ]
  },
  {
    "id": "closing",
    "chapter": "Evidence and next steps",
    "duration": "0:40",
    "variant": "hero",
    "kicker": "23 · The claim you can defend",
    "title": "It observes.\nIt explains.\nIt learns from review.",
    "body": "The working demonstration connects visible evidence to traceable control. The learning lab makes the next experiment explicit.",
    "takeaway": "Trust should come from the right evaluation, not from a moving light or a confident percentage.",
    "notes": [
      "Closing: “I can demonstrate an intersection that reacts to screenshot measurements, explain its score and phase transitions, analyze real photographs with a pretrained detector, and record human choices for supervised learning.”",
      "Then give the measured limitation: “The public detector benchmark reached about 81% precision but 48% recall. The anomaly classifier still failed to detect the held-out anomalies, and learned light choices need reviewed training labels.”",
      "Finish with the next test, not a broad promise: “I would collect newly labeled signalized recordings, improve the detector, add temporal context, and compare policies on identical traffic before claiming a reliable controller.”",
      "Invite questions about any of three levels: what the camera sees, why a light was requested, or what the experiment actually proves. Use the appendix for detailed formulas, model comparisons and failure cases.",
      "The professional conclusion is a bounded claim supported by code, tests and recorded measurements. There is no need to describe any algorithm as universally best."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      },
      {
        "label": "Measured detector benchmark",
        "href": "/classroom/evidence/PRETRAINED_DETECTOR_EVALUATION.md"
      }
    ]
  },
  {
    "id": "appendix-forecast",
    "chapter": "Appendix",
    "duration": "Optional · 1:30",
    "variant": "content",
    "kicker": "A1 · If asked about the project name",
    "appendix": true,
    "title": "Where does the LSTM fit?",
    "body": "The LSTM forecasts hourly demand from historical sequences. It is a separate learning problem from seeing objects or choosing a light.",
    "tables": [
      {
        "caption": "Motorway v2 retrospective test · next-hour MAE, vehicles/hour",
        "headers": [
          "Candidate",
          "MAE",
          "Interpretation"
        ],
        "rows": [
          [
            "XGBoost",
            "153.84",
            "Selected using validation MAE"
          ],
          [
            "LSTM",
            "277.98",
            "Sequential forecasting candidate"
          ],
          [
            "Last observed hour",
            "585.33",
            "Simple persistence baseline"
          ]
        ]
      }
    ],
    "takeaway": "Use the model appropriate to the task; the available evidence does not establish an LSTM advantage.",
    "notes": [
      "An LSTM processes an ordered sequence while carrying a cell state. Input, forget and output gates determine what new information to retain, what earlier information to discard, and what state to expose.",
      "This makes it a reasonable candidate when previous hours carry information about the next hour. It does not make it automatically superior to a tree model using seasonal lags and calendar information.",
      "The v2 report selects XGBoost using validation mean absolute error. The table shows retrospective motorway test errors. Lower MAE is better; the unit is vehicles per hour, not classification accuracy.",
      "The practical XGBoost candidate has longer seasonal lags and target-clock features, so this is not an architecture-only experiment with identical representations. The report also describes fixed budgets, seed and historical-data limitations.",
      "The educational historical forecast panel retains an older trained forecasting model and introspection path. Do not claim its displayed weights are the v2 candidate in this table.",
      "The key explanation for the class is that forecasting hourly motorway flow, detecting people in photographs, and recommending an intersection action require different targets and evaluation methods."
    ],
    "sources": [
      {
        "label": "Hourly forecast evidence",
        "href": "/classroom/evidence/REPORT.md"
      },
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      }
    ]
  },
  {
    "id": "appendix-legacy",
    "chapter": "Appendix",
    "duration": "Optional · 1:00",
    "appendix": true,
    "variant": "dark-metric",
    "kicker": "Historical forecasting · archived comparison",
    "title": "The LSTM was studied, not declared the winner.",
    "body": "Legacy motorway MAE favored XGBoost. Legacy scores are exploratory: earlier data preparation and repeated evaluation limit the conclusions.",
    "tables": [
      {
        "caption": "Historical Metro Interstate experiment · vehicles/hour · lower MAE is better",
        "headers": [
          "Model",
          "MAE"
        ],
        "rows": [
          [
            "XGBoost + calendar",
            "154.3"
          ],
          [
            "LSTM + calendar",
            "201.4"
          ],
          [
            "LSTM traffic only",
            "228.8"
          ],
          [
            "Naive last-hour",
            "585.6"
          ]
        ]
      }
    ],
    "takeaway": "Choose models using task-specific evaluation, not the architecture’s reputation.",
    "notes": [
      "These archived values explain the project’s forecasting history. They do not measure the current intersection controller, recognition system, or light-choice model.",
      "Historical Bike Sharing MAE was 41.3 for LSTM versus 43.1 for XGBoost with traffic history, and 38.5 versus 38.6 with weather and calendar features. These small differences do not establish an LSTM advantage; LSTM MAPE was worse. Compare error units only within each dataset.",
      "The legacy preprocessing includes validation contamination; some runs bridge missing-time gaps or use future-informed weather fills. Test results also informed historical choices. These limitations prevent a confirmatory general-winner claim.",
      "For new forecasting experiments, reports/REPORT.md and its linked models/v2/<run>/manifest.json packages are the source of truth. The LSTM remains educational because its temporal gates and memory can be inspected."
    ],
    "sources": [
      {
        "label": "Archived forecasting results and limitations",
        "href": "/classroom/evidence/LEGACY_REPORT.md"
      },
      {
        "label": "Current forecasting report",
        "href": "/classroom/evidence/REPORT.md"
      }
    ]
  },
  {
    "id": "appendix-recognition-math",
    "chapter": "Appendix",
    "duration": "Optional · 1:30",
    "variant": "dark-metric",
    "kicker": "A2 · If asked how 81% was calculated",
    "appendix": true,
    "title": "Precision rewards correct detections.\nRecall penalizes missed objects.",
    "tables": [
      {
        "caption": "Actual object-benchmark counts after the declared crowd rule",
        "headers": [
          "Quantity",
          "Count",
          "Calculation"
        ],
        "rows": [
          [
            "Matched detections",
            "1,933",
            "True positives"
          ],
          [
            "Unmatched detections",
            "454",
            "False positives"
          ],
          [
            "Missed labeled objects",
            "2,079",
            "False negatives"
          ],
          [
            "Precision / recall",
            "80.98% / 48.18%",
            "1,933 / 2,387; 1,933 / 4,012"
          ]
        ]
      }
    ],
    "takeaway": "A high-precision detector can still miss many objects; both metrics must be reported.",
    "visual": "metrics",
    "notes": [
      "A predicted box is matched to an unused ground-truth box of the same class when their overlap, measured by intersection over union, reaches 0.5. Matching is performed in confidence order.",
      "Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 is the harmonic mean, 2PR/(P+R), which falls when either precision or recall is low.",
      "The dataset has 4,061 annotations including 49 crowd regions. Crowd regions are excluded from the ordinary recall denominator; 93 predictions overlapping qualifying same-class crowd regions are ignored instead of counted as false positives.",
      "Class performance varies substantially: bus recall was 71.67%, car recall 46.61%, person recall 51.44%, and traffic-light recall 29.49%. The aggregate score can conceal the weakest class.",
      "Raising the score cutoff often removes uncertain detections, which may improve precision while reducing recall; it does not change the detector’s learned representation. Any chosen threshold should be fixed before final evaluation.",
      "These are fixed-threshold precision and recall, not COCO mean average precision. The source report contains every class and the reproduction procedure."
    ],
    "sources": [
      {
        "label": "Measured detector benchmark",
        "href": "/classroom/evidence/PRETRAINED_DETECTOR_EVALUATION.md"
      }
    ]
  },
  {
    "id": "appendix-controls",
    "chapter": "Appendix",
    "duration": "Optional · 1:00",
    "variant": "content",
    "kicker": "A3 · Presenter recovery card",
    "appendix": true,
    "title": "What each control changes.",
    "tables": [
      {
        "caption": "Use these explanations if a classmate asks to try the controls",
        "headers": [
          "Control",
          "Effect",
          "Expected observation"
        ],
        "rows": [
          [
            "Sliders + Spawn traffic",
            "Add selected amounts",
            "New visible road users"
          ],
          [
            "Create traffic jam",
            "Queue the chosen approach",
            "Demand and preview change"
          ],
          [
            "Crosswalk / axis request",
            "Queue a request",
            "Reason appears when applied"
          ],
          [
            "Pause / Camera off",
            "Freeze clock / remove input",
            "Frozen scene / clearance hold"
          ]
        ]
      }
    ],
    "takeaway": "Some controls change the scene; others submit requests. Neither is automatic training.",
    "notes": [
      "Reset clears the current simulation and begins a new teaching episode. It is useful for rehearsal, but restarting similar scenes does not create genuinely new real-world conditions.",
      "Automatic traffic controller off chooses a fixed alternating through cycle. Camera input and clearance still matter. It is not a switch that makes every movement run freely.",
      "An axis request can wait behind pedestrian priority or an overdue direction. The last applied decision log, rather than the current preview, tells you when it took effect.",
      "If the scene seems stuck, check Pause, Camera on, a reviewed all-red hold, crossing occupancy and the current phase. Do not bypass a hold simply to make the demonstration look busy.",
      "If the real-image service is unavailable, show the explicit unavailable state and continue with the browser’s simulated controller and the recorded benchmark report. Do not invent an inference result.",
      "Returning to a slideshow should preserve the narrative. The prepared scenarios are for reproducible explanations; they are not a benchmark proving traffic efficiency.",
      "The screenshot-driven controller applies green directions and durations after clearance. Operator requests wait for a safe transition. The separate LSTM panel explains a historical forecast; that forecast does not control these lights. The application is not connected to city signals."
    ],
    "sources": [
      {
        "label": "Live controller implementation",
        "href": "/classroom/evidence/LIVE_INTERSECTION.md"
      },
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      }
    ]
  },
  {
    "id": "appendix-questions",
    "chapter": "Appendix",
    "duration": "Optional · discussion",
    "variant": "content",
    "kicker": "A4 · Questions to be ready for",
    "appendix": true,
    "title": "The difficult questions have precise answers.",
    "bullets": [
      "“Is it trained?” The visual models are pretrained; anomaly fitting is complete; action training needs your labels.",
      "“Is 81% reliable?” It is measured detection precision on the declared public subset.",
      "“Why this algorithm?” It is an efficient, reproducible baseline to test—not a proven universal winner.",
      "“What is missing?” Target-domain labels, sequence context and independent outcome validation."
    ],
    "takeaway": "Name the component, the evidence and the limitation in the same answer.",
    "notes": [
      "“Can it recognize jaywalking or red-light violations?” Not reliably as implemented. Object detection provides locations and classes; movement-dependent violations also require trajectories, signal state, appropriate rules and verified violation labels.",
      "The publisher roundabout gallery remains recognition inspection only. Saved real uploads and video frames can receive explicit human anomaly judgements, including uncertainty, separately from light-action preferences. These controls do not correct bounding boxes; saving a review does not retrain a model. Publisher labels are not edited.",
      "“Does it learn while cars move?” The live screenshot controller follows fixed rules. The learner changes its classifier weights only after human examples are explicitly saved and training is requested. A live operator override is not a training label.",
      "“Why not use a huge neural model?” A more complex model might help with sufficient representative labels and evaluation. Here, frozen visual features plus a small classifier offer a tractable baseline; complexity alone does not solve missing action labels or weak domain coverage.",
      "“Does confidence mean safe?” No. The calibrated action probability is about agreement with human preferences, and even that must be tested. It is not a probability of legal, safe or optimal control.",
      "“What would make the project stronger?” New signalized recordings with reviewed boxes and actions, coherent labeling rules, unseen-camera evaluation, temporal features, then controlled comparisons of traffic outcomes with the same demand."
    ],
    "sources": [
      {
        "label": "Decision training guide",
        "href": "/classroom/evidence/DECISION_TRAINING_GUIDE.md"
      },
      {
        "label": "Measured detector benchmark",
        "href": "/classroom/evidence/PRETRAINED_DETECTOR_EVALUATION.md"
      },
      {
        "label": "Expanded anomaly experiment",
        "href": "/classroom/evidence/STREET_VISION_EXPANDED_EVALUATION.md"
      }
    ]
  }
]
