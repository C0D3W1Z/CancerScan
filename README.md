## BEFORE FORKING AND LOADING THE MODEL, PLEASE DO THE FOLLOWING:

- Download the model file from: https://drive.google.com/file/d/1WvhQhTeAny2Xz77786yQjFvLzfZp8ZDI/view?usp=sharing
- Place the file into "CancerScan/SkinCancerDetector/model/skincancermodel/"

## Voiceflow Links (app needs to be running, .vf files included in repository):

-Chat Bot: https://creator.voiceflow.com/prototype/644e298ed0125e2d5e52ee90

-Lung Cancer Detection: https://creator.voiceflow.com/prototype/644dad2d3e87a3cf4f4f3379

-Mental Training: https://creator.voiceflow.com/prototype/644dc245d0125e2d5e52ee1c

-https://creator.voiceflow.com/prototype/644dcb469e62b46d9a1d7666

## Inspiration

Cancer, the second leading cause of death in the United States, is often viewed as a daunting and overwhelming topic. Cancer can affect anyone, regardless of their age, gender, or background. In our current state, it is crucial that we work towards reducing the burden of cancer on individuals, families, and communities. As such, it is imperative that we increase access to preventative cancer screenings and resources for those living with cancer. Our inspiration for developing CancerScan stems from our fundamental belief that everyone deserves access to the tools necessary to take control of their own health and well-being. Furthermore, we were inspired by the knowledge that early detection is paramount in the fight against cancer. By providing an easy-to-use and accessible platform for cancer screening, we hope to empower individuals to take proactive measures towards their health and potentially detect any issues before they escalate. Additionally, we were also inspired by CancerScan's potential to address healthcare disparities and bridge gaps in access to care for underserved communities that may face greater barriers to screening and care. Given the significant positive impact that a platform for cancer detection and management could have on countless lives, we developed CancerScan with the goal of making a real difference in people's lives and contributing to a healthier and more equitable future.

## What it does
	
CancerScan is an artificial intelligence powered, comprehensive cancer detection, identification, and management platform. Our platform combines cutting-edge technology with a user-friendly interface to deliver a comprehensive and effective solution that is accessible to everyone. We have utilized a machine learning model along with computer vision to identify risk for, presence of, and type of skin cancer from images of skin. We have also utilized a machine learning model to identify the risk for and presence of lung cancer. In addition, we have utilized an LLM in the form of GPT-4 to provide personalized physical exercise guidance, personalized mental training, and personalized information and support to users regarding their condition. Finally, we have used blockchain and Web3 technology with Verbwire to create an achievement and rewards system in the form of NFTs. Here are some features of CancerScan:

Cancerscan offers the following features:

1. Intuitive and User-Friendly Dashboard: Our dashboard is designed to be intuitive and user-friendly, with the goal of making the helpful features of CancerScan accessible to anyone, no matter what their technical background is.
2. Skin Cancer Identification with Computer Vision: Our advanced machine learning model coupled with computer vision technology allows us to accurately identify the risk a user has of contracting or having contracted skin cancer, allowing users to detect skin cancer early and seek treatment as soon as possible.
3. Lung Cancer Identification with Machine Learning: Our machine learning model uses input from users in the form of answers to certain questions to identify the risk of lung cancer in our users. This feature allows for early detection and timely treatment of lung cancer.
4. Personalized AI-Powered Chatbot: Our AI and LLM powered chatbot is designed to provide users with personalized information about their condition.
5. Mental Training System: Our AI powered mental training system provides users with helpful and personalized mental exercises to help with successful treatment and quick recovery.
6. Physical Exercise Guidance System: Our physical exercise guidance system is designed to help users stay in shape for better treatment outcomes and improved overall health by providing users with personalized exercise plans.
7. Web3 Integrated NFT Achievement System: Our Web3 integrated NFT achievement system motivates users to use CancerScan to stay on top of cancer screenings and treatment. Users earn rewards for completing cancer skin and lung cancer identification scans and participating in mental and physical exercises.
8. Encrypted Storage of Account Info: We prioritize user security by encrypting all sensitive account info using Argon2 and PassLib and utilizing a secure SQL database system for storage of user info. This ensures that all sensitive user data is secure and protected.

## How we built it

CancerScan was built utilizing a wide variety of languages and technologies. We used languages such as Python, HTML, CSS, SQL, and JavaScript and technologies such as Flask, AJAX, TensorFlow, Keras, OpenAI, AutoGluon, HashLib, Argon2, in the development of CancerScan. HTML, CSS, and JS were used for the front end and Flask was used for the backend, with AJAX being used for communication between the front and back ends. For our authentication system, we used an SQL database to store user data and HashLib along with Argon2 to hash sensitive user information, ensuring security. We trained a multi-modal classification model utilizing the AutoGluon library, using about 10,000 images of skin cancer from the “Skin Cancer HAM10000” dataset to determine the presence and probability of skin cancer from a single image taken by the user. We also trained a binary classification model to predict whether a person has lung cancer or not based on various features. We used TensorFlow and Keras to train this model with a single dense layer with a sigmoid activation function to output the binary classification results. The model was compiled with the Adam optimizer and a binary cross-entropy loss function, and was trained for about 300 epochs. Through the power of these models, CancerScan is able to predict a user’s risk of contracting skin cancer and lung cancer through just a picture of their skin and their responses to several questions, anywhere and anytime, at no cost to the user. In addition, we used LLMs with VoiceFlow prompt chaining along with their API to gather user data to make accurate lung cancer predictions. We also used VoiceFlow’s GPT-4 functionality, prompt chaining, and API to generate helpful mental and physical exercises for users suffering from cancer, increasing their chances of being successfully treated and helping them make quicker recoveries. Finally, we used VoiceFlow’s GPT-4 functionality, prompt chaining, and API to create a chatbot for those suffering from cancer, helping them learn more about their condition and treatment options. CancerScan also includes a rewards system which rewards users with NFTs which they can purchase using “points”. “Points” are earned when users do certain tasks such as taking cancer identification tests and mental/physical training. We implemented our NFT rewards system through Verbwire’s API, which let us mint NFTs on the Goerli Ethereum test network. We chose to create a rewards system so that users of CancerScan have one more reason to stay motivated when improving their mental and physical health on our platform, increasing the effectiveness of CancerScan.

## Challenges we ran into

Throughout the development of CancerScan, we ran into quite a few challenges. Training the machine learning models was a challenge as finding datasets, figuring out how to effectively preprocess data, and actually training models that require larger amounts of computational power were all very time consuming steps in the creation of our final models. Figuring out how VoiceFlow works and how it should be implemented into CancerScan was also a big challenge that we faced in the creation of our platform. Finally, the setup of our NFT achievement system was also a significant challenge. While Verbwire and its API made setting up our rewards system significantly easier, figuring out how to create our NFTs and implement them into CancerScan was a significant challenge that we faced. Overall, we faced a lot of obstacles that we had to overcome in our development of CancerScan, however we learned a lot from these challenges, making the struggle worth it.

## Accomplishments that we're proud of

While developing CancerScan, we’ve accomplished a few things that we are proud of. The first thing that we accomplished was building our two accurate machine learning models to predict skin and lung cancer, which was a pretty time consuming and difficult task. Our machine learning models started off as pretty inaccurate however we were able to raise both of their accuracies to over ninety percent, which is an accomplishment that we are definitely proud of. Another accomplishment was implementing a Web3 based NFT achievement and rewards system. This was our first time using Web3 to create NFTs, so we are proud of how we were able to create and implement a fully featured NFT achievement system on the Goerli Ethereum testnet. One final accomplishment that we are proud of was effectively implementing Voiceflow and its GPT-4 AI feature. This was our first time using Voiceflow, but it made integrating a LLM with prompt chaining into our product much easier than would have been the case without Voiceflow, making learning about Voiceflow a definite accomplishment as it will serve as a useful tool in the future. Overall, we have made many accomplishments that we are proud of in the development of CancerScan.

## What we learned

We learned a lot of things during the development of CancerScan. We learned a lot about building, tuning, and implementing different types of machine learning models for accuracy. We also learned a lot about Voiceflow and how its LLM and API functionality can be used to build useful applications leveraging powerful AI and machine learning technology. In addition, we also learned a lot about Web3, blockchain, and NFT technology and their implementation and practical uses. Finally, we learned a lot more about AJAX and Flask as CancerScanner required sending a lot of data between the front and back ends to allow its functionality. Overall, PromptHacks was a learning experience for us.

## What's next for CancerScan

We are ambitious about the future of CancerScan. We plan to improve and expand CancerScan’s cancer detecting abilities by adding more machine learning models that will detect other types of cancer in addition to skin and lung cancer. We also plan to make CancerScan’s cancer detecting machine learning models more effective and powerful by taking into consideration a wider range of data to make more informed risk predictions. We also plan to create a mobile app for CancerScan. This will allow existing users to use our platform on the go and will make CancerScan accessible to more users. We’ll also migrate our NFT rewards system from the Goerli Ethereum testnet to the main net as we intend to bring our NFT rewards system out of testing and into actual deployment. Finally, we’ll be using the blockchain to securely store user data to keep it safe, private, and decentralized. Overall, we believe that CancerScan has a lot of potential. 
