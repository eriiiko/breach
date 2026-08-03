
Breach:
breach får nog vänta tills efter presentationen, men jag har änåd några tankar om vad som gått snettjust nu och en idé.
punk1. Varför har vi inte jobbat med den mest naturliga punkten T_kelvin = T0 + 2*T_game (jag tror detta är formeln)
min fråga:Varförbestämmer vi den på förhand istället för att bara hitta ettförlopp som vi gillar, sedan passar vi funktionen
T_kelvin(T_game) = T0 + k * T_game, T0 är precis som innan rumstempretaur, eller T_ambient, k anpassas så att vi får exakt den eld-temperatur som vi vill.

en sak till som bör sägas: Vi har siktat in oss på realistiska konstanter och parametrar. Men jag undrar om vi tänkt lite felpå en punkt:
när vi anger eldtemperatur,t.ex.att enflamma ska vara ca 1000 K, ja, det betyder ju inte nödändigtvis att medeltemperaturen i en tile är 1000K, den bör ju vara mycket mindre, i verkligheten så tar själva eldlågorna upp en lite mindre del.
Så inte konstigt när vi anpassat vår modell att en hel tile på 1/3*1/3*2.5 m3 är 1000K, vilket ledertillatt  luften expanderar något förfärligt.
Verkar som luften expanderar för mycket - om anledningen är precis att vi överskattar medeltemperaturen i en "cell" eller en tile, ellr omdet finns nadra anlednignar,det vågar jag inte säga med säkerhet, men detta kan vara en anledning.
Detta get oss en anledning att försöka sikta på "lägre ingame temps" och sedan skala om temperatur-skalan, eller svartkroppsstrålningsskalan, eller göra något annat- t.ex behålla svartkroppsskalan,menlägga till lite flammamor "on top"
det som börstyra detta är hur snyggt det ser ut. Jag har hela tiden tyckt det verkarsom att luften expanderar lite för mycket - det kan vara så att den expanderat korrekt, men att felet var att vi gett en för stor volym en för hög temperatur.

En sak till - det finns en till spak vi kan justera för att få bukt med de våldsamma svängningarna- det ska finnas möjlighet till damping i luften. Behöver kolla upp exakt hurdet funkar.

Jag börjar även ifågasätta om vi skaha O2 från radie 2 eller radie 1, det kommer an på hur mycket det hela ska kosta beräkningsmässigt. 

Jag tror åter igen att vi ska lämna gridsearch för parametrar, de parametrar som valdes verkade inte bra, glömt vad de heter, men k_fire_increase och decrease hade så vitt olika storleksordningar,jagundrarom det verkligen ärrätt.
Jag tror vi ska försöka bestämma parameterarna i ett verklighetstroget scenario, en efter en.
Vi bör plotta intensitet och temp igen- O2 i rummet. 
Vi bör också hitta ett bra sätt att se hur mycket atmosfären håller på och "stormar".