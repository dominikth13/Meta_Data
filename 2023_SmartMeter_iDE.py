# -*- coding: utf-8 -*-
"""
Created for Metaheuristiken und Data Science (Sommersemester 2023)

basierend auf dem in der Masterarbeit (2022)
"Automatisches Clustering von Stromkunden durch metaheuristische
Clustering Algorithmen unter Verwendung von Smart Meter Daten"
von B. Eichentopf entwickelten Code

"""

# Implementierung der Improved Differential Evolution

#Implementierung der Improved Differential Evolution in Anlehnung an Kuo und Zulvia (2019). Das Paper ist unter folgendem Link abrufbar:
#https://link.springer.com/article/10.1007/s00500-018-3496-z

#Die vorgenommenen Anpassungen basieren auf der Arbeit von Das et al. (2008), welche unter folgendem Link abrufbar ist:
#Das, S.; Abraham, A.; Konar, A. (2008): Automatic clustering using an improved differential evolution algorithm.
#http://03.softcomputing.net/smca-paper1.pdf

## Bibliotheken und Daten laden

import pandas as pd
import numpy as np
import random as rand
from scipy.spatial import distance
import copy
import operator
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
import time
pd.options.mode.chained_assignment = None

#print(pd.__version__)

# Einlesen der jeweiligen analytischen Datenquelle je nach gewähltem Setting
#TODO: hier Anpassen der verwendeten Instanz (aktuell verwendet: 1 Monat, 1000 IDs)
load_df = pd.read_csv("data/setting_1month_1000ids.csv")
del load_df["Unnamed: 0"]
id_df = load_df["ID"]
del load_df["ID"]

X = load_df.to_numpy()

print(X.shape)
feature_count = X.shape[1]

## Global Functions "

"""ANPASSUNG: Mindestanzahl an Datenpunkten innerhalb eines Clusters (in Anlehnung an Das et al. 2008, S. 224)"""
### Zentren eines Chromosoms werden neu initialisiert, wenn einem Cluster weniger als 2 Datenpunkte zugeordnet sind
def reinitializeRandomly(chromosome, partition):
    print(">> reinitializeRandomly", end=", ")
    
    # Auslesen der aktiven Zentren und Bestimmung der resultierenden Datenpunkte pro Cluster
    activeClusters = list(chromosome[chromosome["Active"]<0.50].index)    
    cNumber = len(activeClusters)
    pointsPerCluster = int(len(partition)/cNumber)
    datapointsIndex = list(partition.index)
    
    # zufällige Zuteilung der Datenpunkte zu den aktiven Clustern
    for i in range(0, cNumber):
        datapoints = copy.deepcopy(partition)
        
        # falls bei der Aufteilung der Daten ein Rest entsteht, werden die übrigen Datenpunkte (vor der letzten Zuteilung)
        # im Verhältnis 50:50 aufgeteilt, damit das "Rest"-Cluster nicht zu klein wird
        if ((len(datapointsIndex)-pointsPerCluster)< pointsPerCluster):
            half = int(len(datapointsIndex)/2)
            firstHalf = datapointsIndex[:half]
            lastHalf = datapointsIndex[half:]
            partition.loc[partition.index.isin(firstHalf), "Cluster"] = activeClusters[i-1]
            partition.loc[partition.index.isin(lastHalf), "Cluster"] = activeClusters[i]
        
        # Standard-Zuteilung von der definierten Anzahl an Datenpunkten zu den aktiven Clustern
        else:
            points = rand.sample(datapointsIndex, pointsPerCluster)
            datapointsIndex = [x for x in datapointsIndex if x not in points]
            partition.loc[partition.index.isin(points), "Cluster"] = activeClusters[i]
    
    # Berechnung der neuen Clusterzentren über den Mittelwert der enthaltenen Datenpunkte
    for c in activeClusters:
        data = copy.deepcopy(partition)
        clusterData = copy.deepcopy(data[data["Cluster"]==c])
        clusterData = clusterData.iloc[:,:feature_count]
        chromosome.at[c, "Centroid"] = np.around(list(clusterData.mean(axis=0).values), decimals = 2)
         
    return chromosome, partition

"""ANPASSUNG: Mindestanzahl von 2 aktiven Clustern (in Anlehnung an Das et al. 2008, S. 223)"""
### Zufällige Aktivierung von Clustern, wenn <2 aktive Cluster vorhanden sind
def activateClusters(chromosome, toActivate):
    print(">> activateClusters", end=", ")
    
    chromosome = copy.deepcopy(chromosome)
    count = toActivate
    
    # es werden so viele Zentren zufällig aktiviert, bis wieder 2 aktive Cluster existieren
    for i in range(0,count):
        inactiveClusterIndex = chromosome[chromosome["Active"]>=0.5].index
        threshold = rand.choice(inactiveClusterIndex)
        
        # Aktivierungsstatus des ausgewählten Zentrums wird zufällig im aktiven Intervall bestimmt 
        active = round(rand.uniform(0,0.49),2)
        chromosome.at[threshold, "Active"] = active
    
    activeClusters = chromosome[chromosome["Active"]<0.5]
    
    return chromosome, activeClusters

"""ANPASSUNG: Neu-Zuordnung der Datenpunkte nach Veränderung der Clusterzentren (in Anlehnung an Das et al. 2008, S. 224)"""
### Zuordnung jedes Datenpunktes zu einem aktiven Clusterzentrum auf Basis der Distanzen
def clusterAssigning(chromosome, partition):
    print(">> clusterAssigning", end=", ")
    
    activeClusters = chromosome[chromosome["Active"]<0.50]
    
    # Sicherstellung, dass mindestens 2 Zentren aktiv sind, bevor Zuteilung erfolgt; je nach Anzahl aktiver Cluster
    # Aktivierung von null, ein oder zwei Zentren
    if (activeClusters.empty):
        toActivate = 2
        chromosome, activeClusters = activateClusters(chromosome, toActivate)
    
    elif (len(activeClusters)==1):
        toActivate = 1
        chromosome, activeClusters = activateClusters(chromosome, toActivate)

    # neue aktive Zentren auslesen
    activeClustersIndex = list(chromosome[chromosome["Active"]<0.50].index)
    activeClustersNo = chromosome.loc[chromosome["Active"]<0.50, "Cluster"].unique()
    
    # Zuordnung des Datenpunktes zu dem Zentrum mit der geringsten Distanz
    for i in range(0, len(partition)):
        datapoints = copy.deepcopy(partition)
        datapoints = datapoints.iloc[:,:feature_count]
        distList = []
        
        for j in activeClustersIndex:
            dist = distance.euclidean(datapoints.iloc[i], chromosome.loc[j, "Centroid"])
            distList.append(dist)
        
        assignedCluster = activeClustersNo[distList.index(min(distList))]
        
        partition.at[i, "Cluster"] = assignedCluster
    
    """ANPASSUNG: Mindestanzahl an Datenpunkten innerhalb eines Clusters (in Anlehnung an Das et al. 2008, S. 224)"""
    # Überprüfung, ob mehr als 1 Datenpunkt im Cluster enthalten ist
    for i in activeClustersNo:
        datapoints = partition[partition["Cluster"]==i]
        if (datapoints.shape[0] < 2):
            chromosome, partition = reinitializeRandomly(chromosome, partition)
            
    return chromosome, partition

# Wenn die Learning Rate, Crossover Rate oder der Activation Status innerhalb eines Chromosoms außerhalb der definierten
# Grenzen [0,1] liegt, wird der Wert an die jeweilige Grenze angepasst (in Anlehnung an Das et al. 2008)

def roundToInterval(parameter):
    
    if (parameter < 0):
        parameter = 0.0
    elif (parameter > 1):
        parameter = 1.0
    
    return parameter

## Creating the Initial Population

### Applying the Cluster Decomposition Algorithm

### Durchführung des Cluster Decomposition Algorithm 
def clusterDecomposition(dataset, radius):
    print(">> clusterDecomposition")
    
    cNumber = 0
    
    # Hilfs-Dataframe, um zu prüfen, ob alle Datenpunkte einem Cluster zugeteilt wurden
    data = copy.deepcopy(dataset)
    # Dataframe, in dem jedem Datenpunkt ein Cluster zugeordnet wird
    datasetClustered = copy.deepcopy(dataset)
    
    # So lange wie im Set noch Datenpunkte enthalten sind, werden diese den jeweiligen Clustern zugeteilt
    while (not data.empty):      
        print("x", end=", ")
        # Auswahl eines zufälligen Datenpunktes als Clusterzentrum
        clusterCenter = data.sample() 
        testClusterCenter = clusterCenter.values.flatten()
        for i in range(len(dataset)):
            # Datenpunkt überspringen, wenn er nicht mehr im Set enthalten ist, da er bereits zugeordnet ist
            if not i in data.index.values: 
                pass
            # Ausgewähltes Clusterzentrum wird aus Set entfernt, da somit einem Cluster zugeordnet
            elif i==clusterCenter.index.item():
                datasetClustered.at[i, "Cluster"] = cNumber
                data = data.drop(clusterCenter.index.item())
            else:
                # ggf. Zuordnung der übrigen Datenpunkte zum Cluster j über euklidische Distanz
                neighbor = pd.DataFrame(dataset.loc[i]).transpose()
                testNeighbor = neighbor.values.flatten()
                dist = distance.euclidean(testClusterCenter, testNeighbor)
                #dist = distance.euclidean(clusterCenter, neighbor)
                # Datenpunkt wird nur zugeordnet, wenn er innerhalb des vorgegebenen Radius zum Zentrum ist
                if (dist < radius):
                    datasetClustered.at[i, "Cluster"] = cNumber
                    data = data.drop(neighbor.index.item())       
        # Clusternummer wird erhöht, wenn noch Datenpunkte einem neuen Cluster zugeordnet werden müssen
        if (not data.empty):
            cNumber += 1
    print("Done.")
     
    # wenn alle Datenpunkte einem Cluster zugeordnet wurden, wird die Clusteranzahl des Individuum gespeichert
    for i in range(len(datasetClustered)):
        datasetClustered.at[i, "Anzahl Cluster"] = cNumber+1

    return datasetClustered

### Creating Chromosomes & Evaluating their Fitness

# Erstellung der initialen Chromosome auf Basis der Clusterzuteilungen des Cluster Decomposition Algorithm
def createChromosome(partition, cMax, individual):
    print(">> createChromosome", end=", ")
    
    # Erstellung eines Chromosom-DF
    chromosome = pd.DataFrame(columns=["Ind","LR","CR","Cluster","Active","Centroid"])
    
    ### 1. TEIL: LEARNING RATE & CROSSOVER RATE
    learningRate = rand.random()
    crossoverRate = rand.random()
    
    # Einfügen der bisherigen Informationen in das Chromosom
    for i in range(0,int(cMax)):
        chromosome.loc[i, "Ind"] = individual
        chromosome.loc[i, "Cluster"] = i
        chromosome.loc[i, "LR"] = learningRate
        chromosome.loc[i, "CR"] = crossoverRate
        
    ### 2.& 3. TEIL: ACTIVATION THRESHOLD & CENTROIDS
    for i in range(0,int(cMax)):
        
        cNumber = partition.loc[i, "Anzahl Cluster"]
        
        if i < (cNumber):
            # 2.Teil: Activation Threshold
            chromosome.loc[i, "Active"] = 1
            
            # 3.Teil: Centroids
            clusterData = copy.deepcopy(partition[partition["Cluster"]==i])
            clusterData = clusterData.iloc[:,:feature_count]
            chromosome.loc[i, "Centroid"] = np.around(list(clusterData.mean(axis=0).values), decimals=2)
            
        else:
            # 2.Teil: Activation Threshold
            chromosome.loc[i, "Active"] = 0
            
            # 3.Teil: Centroids
            clusterData = copy.deepcopy(partition)
            clusterData = clusterData.iloc[:,:feature_count]
            lowerBound = list(clusterData.min(axis=0).values)
            upperBound = list(clusterData.max(axis=0).values)
            
            randomCentroidList = []
            
            for j in range(0,len(lowerBound)):
                centroid = round(rand.uniform(lowerBound[j],upperBound[j]),2)
                randomCentroidList.append(centroid)
            
            chromosome.loc[i, "Centroid"] = randomCentroidList
            
    return chromosome


#TODO:  weitere Fitnessbewertungen basierend z.B. auf anderen Cluster Validation Indizes verwenden und testen
#       dazu ggf. eigene Funktion(en) schreiben
#TODO:  im Idealfall kombinierte Fitnessbewertung basierend auf mehreren Indizes verwenden (sofern sinnvoll)
"""ANPASSUNG: Evaluation nur auf Basis der aktiven Zentren (in Anlehnung an Das et al. 2008, S. 224)"""
def evaluateFitness(chromosome, partition):
    print(">> evaluateFitness", end=", ")
    
    # Überprüfung, ob mindestens 2 Cluster aktiv sind, ansonsten werden zufällige Cluster aktiviert
    activeClusters = chromosome[chromosome["Active"]<0.50].reset_index(drop=True)
    
    validationIndex = calculate_vi(activeClusters, chromosome, partition)
    ssc = calculate_ssc(activeClusters, chromosome, partition)

    # Zuordnung des Fitnesswertes zum Chromosom
    chromosomeFitness = copy.deepcopy(chromosome)
    for i in range(0, len(chromosome)):
        chromosomeFitness["SSC"] = ssc
        chromosomeFitness["VI"] = validationIndex
        if False: #Hier entscheiden, welches Validierungsmaß verwendet werden soll
            chromosomeFitness["Fitness"] = validationIndex
        else:
            # Vorzeichen muss getauscht werden, da bei SSC gilt Min = -1 < 1 = Max, aber wir brauchen Max = -1 < 1 = Min
            chromosomeFitness["Fitness"] = ssc * (-1)
    
    return chromosomeFitness, partition

def calculate_vi(activeClusters, chromosome, partition):
    ### INTRA CLUSTER: ABSTAND DER DATENPUNKTE INNERHALB EINES CLUSTERS (KOMPAKTHEIT)
    intraCluster = 0

    # für jedes Cluster soll die Distanz zu den enthaltenen Datenpunkten bestimmt werden
    for i in range(0, len(activeClusters["Cluster"])):
        
        # Auslesen des Centroids des jeweiligen Clusters und Liste der aktiven Cluster
        clusterList = activeClusters["Cluster"].unique()
        centroid = activeClusters.loc[i, "Centroid"]
            
        # Bestimmung der Datenpunkte, die dem Cluster zugeordnet sind
        datapoints = copy.deepcopy(partition[partition["Cluster"]==clusterList[i]]).reset_index(drop=True)
        datapoints = datapoints.iloc[:,:feature_count]

        # für jeden Datenpunkt des Clusters wird die Distanz zum Zentrum berechnet und zu intraCluster addiert
        for j in range(0, len(datapoints)):

            dist = distance.euclidean(datapoints.iloc[j], centroid)
            intraCluster += dist
                
    intraCluster = intraCluster/len(partition)
    
    ### INTER CLUSTER: ABSTAND ZU ANDEREN ZENTREN (hier: Minimale Distanz zwischen zwei Clustern)
    distanceCentroids = []
    for i in range(0, len(activeClusters["Cluster"])):
        for j in range(0, len(activeClusters["Cluster"])):
            
            if j == i: continue
            else:
                dist = distance.euclidean(activeClusters.loc[i, "Centroid"], activeClusters.loc[j, "Centroid"])
                distanceCentroids.append(dist)

    interCluster = min(distanceCentroids)
        
    # BERECHNUNG DES FINALEN VI NACH TURI (2001)
    #TODO: ggf. Anpassen der Parameter für die Berechnung des VI nach Turi
    cNumber = len(activeClusters["Cluster"])
    alpha = 30
    mü = 2
    sigma = 1
    normalDistr = norm.pdf(cNumber, mü, sigma)
        
    return round((alpha*normalDistr+1)*(intraCluster/interCluster),4)

def calculate_ssc(activeClusters, chromosome, partition):
    clusterList = activeClusters["Cluster"].unique()
    datapoints = copy.deepcopy(partition[partition["Cluster"].isin(clusterList)]).reset_index(drop=True)
    datapoints = datapoints.iloc[:,:feature_count]
    ssc = metrics.silhouette_score(X=datapoints.to_numpy(), labels=copy.deepcopy(partition[partition["Cluster"].isin(clusterList)])["Cluster"].reset_index(drop=True))
    return ssc

### Putting everything together & Creating the Initial Population

### Initialisierung
def createInitialPopulation(dataset, popSize, radius):
    print(">> createInitialPopulation")
    
    cMax = 0
    
    # DF zur Speicherung aller entstandenen initialen Individuen
    partitioning = pd.DataFrame()
    # DF zur Speicherung der entstandenen initialen Population
    population = pd.DataFrame(columns=["Ind","LR","CR","Cluster","Active","Centroid", "Fitness","VI","SSC"])
    datasetClusteredList = []
    ### AUFTEILUNG DES DATENSATZES IN EINE INITIALE PARTITIONIERUNG
    for i in range(0, popSize):
        
        # Aufteilung des Datensatzes über Cluster Decomposition Algorithm
        datasetClustered = clusterDecomposition(dataset, radius)
        datasetClustered["Ind"] = i

        # Speichere datasetClustered in Liste, welche nach der Schleife zusammen zu partitioning hinzugefügt wird
        #  Konkatenierung pro Iteration erzeugt exponentiale Laufzeit, siehe https://stackoverflow.com/questions/36489576/why-does-concatenation-of-dataframes-get-exponentially-slower
        datasetClusteredList.append(datasetClustered)
        
        # Bestimmung der maximalen Clusteranzahl der Population
        cNumber = datasetClustered.loc[0, "Anzahl Cluster"]
        
        if cNumber > cMax:
            cMax = cNumber
    
    partitioning = pd.concat(datasetClusteredList)
    del datasetClusteredList
    print("Ende CDA")
    
    print(f"Maximale Clusteranzahl über CDA: {cMax}")
    
    chromosomeFitnessList = []
    ### ERSTELLUNG DER CHROMOSOME AUS DEN ERSTELLTEN INDIVIDUEN UND BEWERTUNG ANHAND DER FITNESSFUNKTION
    for i in range(0, popSize):
        
        partition = partitioning[partitioning["Ind"]==i]
        chromosome = createChromosome(partition, cMax, i)
        
        # Zuordnung der Datenpunkte zu den aktiven Clustern
        chromosome, partition = clusterAssigning(chromosome, partition)
        
        # Bewertung der Fitness des jeweiligen Chromosoms, Einfügen des Chromosoms in die Population, Einfügen der
        # Clusterzuteilung in die Zuteilungen der Population
        #TODO: ggf. Aufruf von evaluateFitness anpassen, basierend auf anderen implementierten Fitnessbewertungen
        chromosomeFitness, partition = evaluateFitness(chromosome, partition)
        #population = population.append(chromosomeFitness)
        chromosomeFitnessList.append(chromosomeFitness)
        partitioning.loc[partitioning.Ind == i,:] = partition
    
    population = pd.concat([population, pd.concat(chromosomeFitnessList)])
    del chromosomeFitnessList
    return population, partitioning

## Optimizing the Initial Solution

### Mutation & Crossover

# Erstellung des Trial Vector auf Basis des Best Solution Effects
def createTrialVector(chromosome, neighbor, pBest):
    print(">> createTrialVector")
    
    trialVector = pd.DataFrame(columns=["Ind","LR","CR","Cluster","Active","Centroid"])
    
    for i in range(0, len(chromosome["Cluster"])):
        # Übernahme der unveränderten Parameter
        trialVector.at[i, "Ind"] = chromosome.loc[i, "Ind"]
        trialVector.at[i, "Cluster"] = chromosome.loc[i, "Cluster"]
        
        # Mutation Learning Rate, Crossover Rate & Activation Threshold
        lRate = chromosome.loc[i, "LR"] + chromosome.loc[i,"LR"]*(pBest.loc[i, "LR"]-neighbor.loc[i,"LR"])
        lRate = roundToInterval(lRate)
        trialVector.at[i, "LR"] = lRate
         
        cRate = chromosome.loc[i, "CR"] + chromosome.loc[i, "LR"]*(pBest.loc[i, "CR"]-neighbor.loc[i,"CR"])
        cRate = roundToInterval(cRate)
        trialVector.at[i, "CR"] = cRate
        
        activation = round((chromosome.loc[i, "Active"] + chromosome.loc[i, "LR"]*(pBest.loc[i, "Active"] - neighbor.loc[i, "Active"])),2)
        activation = roundToInterval(activation)            
        trialVector.at[i, "Active"] = activation
        
        # Mutation der Zentren
        difference = list(map(operator.sub, pBest.loc[i, "Centroid"], neighbor.loc[i, "Centroid"]))
        learningRate = chromosome.loc[i, "LR"]
        differenceLearning = [(x*learningRate) for x in difference]
        trialVector.at[i, "Centroid"] = np.around(list(map(operator.add, chromosome.loc[i, "Centroid"],differenceLearning)), decimals = 2)
        
    return trialVector

# Erstellen des Offsprings aus dem Chromosom und dem Trial Vector
def createOffspring(chromosome, trialVector):
    print(">> createOffspring", end=", ")
    
    offspring = pd.DataFrame(columns=["Ind","LR","CR","Cluster","Active","Centroid", "Fitness"])
    
    # für jedes Cluster des Chromosoms wird einzeln entschieden, ob das Cluster des Trial Vectors genutzt wird oder nicht
    for i in range(0, len(chromosome)):
        randomCR = rand.random()
        
        if (chromosome.loc[i, "CR"] < randomCR):
            offspring.loc[i] = trialVector.loc[i]
            
        else:
            offspring.loc[i] = chromosome.loc[i]
            
            # Um eine konsistente Learning & Crossover Rate zu erhalten, falls ein Teil des Chromosoms verändert wird, 
            # werden die beiden Parameter immer vom Trial Vector übernommen
            offspring.at[i, "LR"] = trialVector.loc[i, "LR"]
            offspring.at[i, "CR"] = trialVector.loc[i, "CR"]

    return offspring

# Erstellung eines komplett neuen Chromosoms als Teil der Saturated Solution
def createRandomChromosome(chromosome, partition):
    print(">> createRandomChromosome", end=", ")
    
    # Erstellung eines Chromosom-DF
    nextChromosome = pd.DataFrame(columns=["Ind","LR","CR","Cluster","Active","Centroid"])
    
    ### 1. TEIL: LEARNING RATE & CROSSOVER RATE
    learningRate = rand.random()
    crossoverRate = rand.random()
    
    # Einfügen der bisherigen Informationen in das Chromosom
    for i in range(0, chromosome.shape[0]):
        nextChromosome.loc[i, "Ind"] = chromosome.loc[i, "Ind"]
        nextChromosome.loc[i, "Cluster"] = i
        nextChromosome.loc[i, "LR"] = learningRate
        nextChromosome.loc[i, "CR"] = crossoverRate
        
        activation = round(rand.random(), 2)
        nextChromosome.loc[i, "Active"] = activation
    
    ### 3.Teil: Centroids - dabei zufällige neue Zentren im Wertebereich der Datenpunkte
        clusterData = copy.deepcopy(partition)
        clusterData = clusterData.iloc[:,:feature_count]
        lowerBound = list(clusterData.min(axis=0).values)
        upperBound = list(clusterData.max(axis=0).values)
            
        randomCentroidList = []
            
        for j in range(0,len(lowerBound)):
            centroid = round(rand.uniform(lowerBound[j],upperBound[j]),2)
            randomCentroidList.append(centroid)
            
        nextChromosome.loc[i, "Centroid"] = randomCentroidList
                
    ### Datenpunkte zuordnen und Fitness bewerten
    nextChromosome, partition = clusterAssigning(nextChromosome, partition)
    #TODO: ggf. Aufruf von evaluateFitness anpassen, basierend auf anderen implementierten Fitnessbewertungen
    nextChromosomeFitness, partition = evaluateFitness(nextChromosome, partition)
             
    return nextChromosomeFitness, partition

# Auswahl des Chromosoms für die nächste Generation auf Basis der definierten Selektionskriterien
def chooseChromosome(chromosome, solutionCandidate, partition, potentialPartition,
                     temp, sat, tempLimit, satLimit):
    print(">> chooseChromosome")
    
    improvement = solutionCandidate["Fitness"].min() - chromosome["Fitness"].min()
    
    ### Handling Down Hill: Solution Candidate hat schlechtere Fitness als Ursprungschromosom
    if (improvement > 0):        
        # Wenn tempLimit noch nicht erreicht, dann Reduzierung von temp
        if (temp > tempLimit):
            temp -= 0.05
            
        # Bestimmung des verwendeten Chromosoms gemäß Temperaturregeln
        if (temp > tempLimit): 
            nextChromosome = copy.deepcopy(solutionCandidate)
            partition = potentialPartition
        else: 
            nextChromosome = copy.deepcopy(chromosome)
        
        # Da eine Veränderung der Lösung stattgefunden hat, wird sat wieder auf 1 gesetzt
        sat = 1
        
    ### Saturated Solution: Fitness beider Kandidaten ist gleich
    elif (improvement == 0):
        # Wenn satLimit noch nicht erreicht, dann Reduzierung von sat
        if (sat > satLimit):
            sat -= 0.05
            
        # Bestimmung des verwendeten Chromosoms gemäß Saturationsregeln
        if (sat > satLimit): 
            nextChromosome = copy.deepcopy(solutionCandidate)
            partition = potentialPartition
        else:
            nextChromosome, partition = createRandomChromosome(chromosome, partition)
    
    ### Solution Candidate verbessert die Lösung
    else:
        nextChromosome = copy.deepcopy(solutionCandidate)
        partition = potentialPartition
        temp = tempLimit
        sat = satLimit
    
    return nextChromosome, temp, sat, partition

### Acceleration

# Erzeugung des Acceleration Vectors im Zuge der Acceleration
def createAccelerationVector(nextGenerationBest, generationBest, pWorst):
    print(">> createAccelerationVector", end=", ")
    
    if (nextGenerationBest["Fitness"].min() < generationBest["Fitness"].min()):
        
        # nextGenerationBest entspricht Acceleration Vector und ersetzt pWorst in der neuen Generation
        # Anpassen der Ind-Nummer, sodass dieses später leichter in die Generation eingefügt werden kann
        for i in range (0, len(pWorst)):
            nextGenerationBest.at[i, "Ind"] = pWorst.loc[i, "Ind"]
            
        accelerationVector = copy.deepcopy(nextGenerationBest)
    
    else:
        
        ### pWorst mutiert in Richtung generationBest und nextGenerationBest
        accelerationVector = pd.DataFrame(columns=["Ind","LR","CR","Cluster","Active","Centroid","Fitness","VI","SSC"])
        nü = 0.3 # Acceleration Rate
            
        for i in range(0, len(nextGenerationBest)):
            # Übernahme der unveränderten Parameter
            accelerationVector.loc[i, "Ind"] = pWorst.loc[i, "Ind"]
            accelerationVector.loc[i, "Cluster"] = pWorst.loc[i, "Cluster"]
            
            # Anpassung Learning Rate, Crossover Rate & Activation Threshold
            lRate = pWorst.loc[i, "LR"] - nü*(nextGenerationBest.loc[i, "LR"] - generationBest.loc[i, "LR"])
            lRate = roundToInterval(lRate)
            accelerationVector.loc[i, "LR"] = lRate
            
            cRate = pWorst.loc[i, "CR"] - nü*(nextGenerationBest.loc[i, "CR"] - generationBest.loc[i, "CR"])
            cRate = roundToInterval(cRate)
            accelerationVector.loc[i, "CR"] = cRate
        
            activation = round((nextGenerationBest.loc[i, "Active"] - 
                nü*(nextGenerationBest.loc[i, "Active"] - generationBest.loc[i, "Active"])),2)
            activation = roundToInterval(activation)
            accelerationVector.loc[i, "Active"] = activation

            # Anpassung Centroids
            difference = list(map(operator.sub, nextGenerationBest.loc[i, "Centroid"], generationBest.loc[i, "Centroid"]))
            differenceLearning = [(x*nü) for x in difference]
            accelerationVector.loc[i, "Centroid"] = np.around(list(map(operator.sub, nextGenerationBest.loc[i, "Centroid"],
                                                                differenceLearning)), decimals = 2)
    
    return accelerationVector

# Durchführung der Acceleration auf Basis des erzeugten Acceleration Vectors
def accelerateExploration(nextGeneration, accelerationVector, pWorst):
    print(">> accelerateExploration")

    newNextGeneration = copy.deepcopy(nextGeneration)
    
    # Entfernen des schlechtesten Individuums aus der Generation
    newNextGeneration = newNextGeneration[newNextGeneration["Ind"] != pWorst["Ind"].mean()]
    
    # Einfügen des Acceleration Vectors in die neue Generation (an der Stelle der schlechtesten Lösung)
    #newNextGeneration = newNextGeneration.append(accelerationVector).sort_values(by= ["Ind","Cluster"])
    newNextGeneration = pd.concat([newNextGeneration,accelerationVector]).sort_values(by= ["Ind","Cluster"])
        
    return newNextGeneration

## Updating the relevant Parameters"

# Finden der besten Lösung einer Population
def findGlobalBest(population):
    print(">> findGlobalBest")
    
    pBest = population[population["Fitness"]==population["Fitness"].min()]
    # Auswahl der ersten besten Lösung der Generation (falls mehrfach vorhanden)
    pBest = pBest[pBest["Ind"]==pBest["Ind"].min()]
    
    return pBest

def updateParameters(population):
    print(">> updateParameters", end=", ")

    # Auswahl der ersten besten Lösung der Generation (falls mehrfach vorhanden)
    populationBest = population[population["Fitness"]==population["Fitness"].min()]
    populationBest = populationBest[populationBest["Ind"]==populationBest["Ind"].min()]
    
    # Auswahl der ersten schlechtesten Lösung der Generation (falls mehrfach vorhanden)
    pWorst = population[population["Fitness"]==population["Fitness"].max()]
    pWorst = pWorst[pWorst["Ind"]==pWorst["Ind"].min()]
    
    return populationBest, pWorst

## Updating the Population

# Erstellung der nächsten Generation
def updatePopulation(popSize, generation, pBest, partitioning, temp, sat, tempLimit, satLimit):
    print(">> updatePopulation")
    
    nextGeneration = pd.DataFrame(columns=["Ind","LR","CR","Cluster","Active","Centroid", "Fitness"])
    partitioning = copy.deepcopy(partitioning)
    generation = copy.deepcopy(generation)
    
    for i in range(0, popSize):
        partition = partitioning[partitioning["Ind"]==i]
        chromosome = generation[generation["Ind"]==i]
            
        # Bestimmung des Nachbarn für Mutation
        neighborID = rand.choice([x for x in range(0, (generation["Ind"].max()+1)) if x != i])
        neighbor = generation[generation["Ind"]== neighborID]
            
        # Erstellung des Trial Vectors unter Verwendung der besten Lösung ("Best Solution Effect")
        trialVector = createTrialVector(chromosome, neighbor, pBest)
            
        # Erstellung des Solution Candidates und potentieller Zuordnung der Datenpunkte, um Fitness mit dem ursprünglichen
        # Chromosom vergleichen zu können
        offspring = createOffspring(chromosome, trialVector)
        offspring, offspringPartition = clusterAssigning(offspring, partition)
        #TODO: ggf. Aufruf von evaluateFitness anpassen, basierend auf anderen implementierten Fitnessbewertungen
        solutionCandidate, potentialPartition = evaluateFitness(offspring, offspringPartition)
            
        # Auswahl des Chromosoms, das für die nächste Generation verwendet wird
        nextChromosome, temp, sat, partition = chooseChromosome(chromosome,solutionCandidate,partition,potentialPartition,
                                                                temp,sat,tempLimit,satLimit)
            
        # Auffüllen der neuen Generation mit nextChromosome
        #nextGeneration = nextGeneration.append(nextChromosome)
        nextGeneration = pd.concat([nextGeneration,nextChromosome])
        partitioning.loc[partitioning.Ind == i,:] = partition
    
    return nextGeneration, temp, sat, partitioning

## Updating the best solution with k-means

# Update der besten gefundenen Lösung über k-means
def updateBestSolution(pBest, generation, partitioning, nextGenerationBest, currentBestPartition):
    print(">> updateBestSolution")
      
    generation = copy.deepcopy(generation)
    partitioning = copy.deepcopy(partitioning)
    pBest = copy.deepcopy(pBest)
    
    # Die aktiven Zentren der aktuellen besten Lösung werden als Startzentren für k-means verwendet
    activeClusters = pBest[pBest["Active"]<0.5]
    k = len(activeClusters.Cluster)
    startCentroids=np.array([np.array(xi) for xi in activeClusters.Centroid])

    # Fitting der Zentren mit k-means
    kmeans = KMeans(init=startCentroids,n_clusters=k,n_init=1, max_iter = 300)
    model = kmeans.fit(X)

    # Angleichen der gefundenen k-means Cluster-Nummern an die der aktiven Zentren (k-means startet bei 0, aber die
    # iDE Zentren nicht zwangsläufig)
    model.labels_ = model.labels_.tolist()
    model.labels_ = [activeClusters.Cluster.unique()[model.labels_[i]] for i in range(0,len(model.labels_))]

    # Einfügen der neuen Zentren in die ursprüngliche beste Lösung
    newGlobalBest = copy.deepcopy(pBest)
    newCentroids = model.cluster_centers_.tolist()
    activeClusters['Centroid'] = [np.around(list(newCentroids[i]),decimals=2) for i in range(0, len(activeClusters['Centroid']))]
    newGlobalBest.update(activeClusters)

    # Update der Zuordnung der Datenpunkte zum Cluster
    newPartition = copy.deepcopy(currentBestPartition)
    for i in range(0, len(newPartition)):
        newPartition.at[i, "Cluster"] = model.labels_[i]

    # Bestimmung der Fitness der neuen Lösung
    #TODO: ggf. Aufruf von evaluateFitness anpassen, basierend auf anderen implementierten Fitnessbewertungen
    newGlobalBestFit, newPartition = evaluateFitness(newGlobalBest, newPartition)

    # Übernehmen der angepassten Lösung als neues pBest
    pBest = copy.deepcopy(newGlobalBestFit)
    currentBestPartition = copy.deepcopy(newPartition)

    return generation, partitioning, pBest, currentBestPartition

## Creating the Improved Differential Evolution Algorithm

def improvedDifferentialEvolution(data, popSize, radius, cycles, iterations, tempLimit, satLimit):
    print(">> improvedDifferentialEvolution")
    # Initialisierung
    #TODO: bei Bedarf ggf. mit anderer Initialisierung experimentieren bzw. Anpassen von createInitialPopulation
    initPop, partitioning = createInitialPopulation(data, popSize, radius)

    # Parameterdeklaration für den Optimierungsprozess
    generation = copy.deepcopy(initPop)
    pBest = findGlobalBest(initPop)
    currentBestPartition = partitioning.loc[partitioning["Ind"]==pBest["Ind"].min(),:]
    temp = 1
    sat = 1
    
    # Optimierung
    for c in range(0, cycles): 
        print(f"Cycle {c}:")
        
        for t in range(0, iterations):
            print(f"Iteration {t}:")

            ## Updating der relevanten Parameter
            generationBest, pWorst = updateParameters(generation)

            ## Updating der Population/Erstellen der neuen Generation für t+1
            nextGeneration, temp, sat, partitioning = updatePopulation(popSize,generation,pBest,
                                                                       partitioning,temp,sat,tempLimit,satLimit)

            # Updating der relevanten Parameter für die neue Generation
            nextGenerationBest, pWorst = updateParameters(nextGeneration)
            # Anpassung von pBest, falls bereits bessere Lösung gefunden
            if (nextGenerationBest["Fitness"].min() < pBest["Fitness"].min()):
                pBest = copy.deepcopy(nextGenerationBest)
                currentBestPartition = copy.deepcopy(partitioning.loc[partitioning["Ind"]==nextGenerationBest["Ind"].min(),:])
            
            # Improving the Exploration Speed ("Acceleration")
            # Erstellung des Acceleration Vectors und ersetzen der schlechtesten Lösung und Zuteilung, um die neue Fitness
            # zu bestimmen
            accelerationVector = createAccelerationVector(nextGenerationBest, generationBest, pWorst)
            partitionWorst = partitioning.loc[partitioning["Ind"]==accelerationVector["Ind"].min(),:]
            accelerationVector, partitionWorst = clusterAssigning(accelerationVector, partitionWorst)
            #TODO: ggf. Aufruf von evaluateFitness anpassen, basierend auf anderen implementierten Fitnessbewertungen
            accelerationVector, partitionWorst = evaluateFitness(accelerationVector, partitionWorst)
            # Ersetzen von pWorst in der neuen Generation
            newNextGeneration = accelerateExploration(nextGeneration, accelerationVector, pWorst)
            partitioning.loc[partitioning.Ind == pWorst.Ind.min(),:] = partitionWorst
            
            # Ersetzen der aktuellen Generation durch die neue erstellte Generation für t+1
            generation = copy.deepcopy(newNextGeneration)

        # Updating der besten gefundenen Lösung über k-means
        generation, partitioning, pBest, currentBestPartition = updateBestSolution(pBest, generation, partitioning, nextGenerationBest, currentBestPartition)
        
        # Zwischenspeichern der insgesamt besten gefundenen Lösung und Zuteilung über alle Generationen hinweg
        if (c==0):
            finalBest = copy.deepcopy(pBest)
            finalBestPartition = copy.deepcopy(currentBestPartition)
        
        if (pBest.Fitness.min() < finalBest.Fitness.min()):
            finalBest = copy.deepcopy(pBest)
            finalBestPartition = copy.deepcopy(currentBestPartition)
    
    return finalBest, finalBestPartition

## Running the Algorithm

### Parameter Settings
data = load_df

#TODO: Parametersettings anpassen und untersuchen

# Initialization (Parameter des VI werden direkt in der entsprechenden Funktion deklariert)
radius = 2

# Evolutionsparameter
popSize = 40
iterations = 20
cycles = 1

# Selektionsparameter
tempLimit = 0.5
satLimit = 0.5

# Durchführung des gesamten Algorithmus 

# Ergebnis-DataFrame zur Zwischenspeicherung der relevanten Informationen
#TODO: ggf. um weitere Validierungskennzahlen ergänzen
results_df = pd.DataFrame(columns=["Clusters","VI","SI","DB","Time","Verteilung"])

# Anzahl der Durchläufe festlegen
for i in range(1,2):
    print(f"Durchlauf: {i}")
    start = time.time()
    pBest, currentBestPartition = improvedDifferentialEvolution(data, popSize, radius, cycles, iterations, tempLimit, satLimit)
    end = time.time() 
    
    # Evaluation
    #TODO: ggf. um weitere Validierungskennzahlen ergänzen
    results_df.loc[i,"Clusters"] = len(sorted(currentBestPartition.Cluster.unique()))
    results_df.loc[i,"VI"] = pBest.VI.min()
    si = metrics.silhouette_score(currentBestPartition.iloc[:,:feature_count].to_numpy(), currentBestPartition["Cluster"])
    results_df.loc[i,"SI"] = pBest.SSC.max()
    db = davies_bouldin_score(currentBestPartition.iloc[:,:feature_count].to_numpy(),currentBestPartition["Cluster"])
    results_df.loc[i,"DB"] = db
    
    # Rechenzeit
    results_df.loc[i, "Time"] = (end-start)/60
    
    # Anzahl an Datenpunkten pro Cluster bestimmen
    activeClusters = list(pBest.loc[pBest["Active"]<0.5,"Cluster"])
    cluster = sorted(currentBestPartition.Cluster.unique())

    verteilung_lst = []
    for j in cluster:
        pointsPerCluster = currentBestPartition[currentBestPartition.Cluster==j]
        verteilung_lst.append(len(pointsPerCluster))
    results_df.loc[i, "Verteilung"] = verteilung_lst

    #Export der Lösung
    filename1 = "partitioning_No" + str(i) + ".csv"
    filename2 = "pBest_No" + str(i) + ".csv"
    currentBestPartition["ID"]=id_df
    currentBestPartition.to_csv(filename1)
    pBest.to_csv(filename2)
    
    print(results_df)

print(results_df)
 