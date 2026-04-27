#Visualizations: line 170; Statistics: line 302; Relative entropy: line 391; Correlations: line 413; Levene's test: line 431
rm(list=ls())

library(ggplot2)
library(lme4)
library(plyr)
library(gridExtra)
library(grid)
library(lattice)
library(reshape2)
library(lmerTest)
library(trimr)
library(effects)
detach(dplyr)

plotPalette <- c("#993300", "#3399cc", "#FF9900", "#669900", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
library(ez)
library(reshape2)
summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  library(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(N    = length2(xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     sd   = sd     (xx[[col]], na.rm=na.rm)
                   )
                 },
                 measurevar
  )
  
  # Rename the "mean" column    
  datac <- rename(datac, c("mean" = measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}

## Summarizes data, handling within-subjects variables by removing inter-subject variability.
## It will still work if there are no within-S variables.
## Gives count, un-normed mean, normed mean (with same between-group mean),
##   standard deviation, standard error of the mean, and confidence interval.
## If there are within-subject variables, calculate adjusted values using method from Morey (2008).
##   data: a data frame.
##   measurevar: the name of a column that contains the variable to be summariezed
##   betweenvars: a vector containing names of columns that are between-subjects variables
##   withinvars: a vector containing names of columns that are within-subjects variables
##   idvar: the name of a column that identifies each subject (or matched subjects)
##   na.rm: a boolean that indicates whether to ignore NA's
##   conf.interval: the percent range of the confidence interval (default is 95%)
summarySEwithin <- function(data=NULL, measurevar, betweenvars=NULL, withinvars=NULL,
                            idvar=NULL, na.rm=FALSE, conf.interval=.95, .drop=TRUE) {
  
  # Ensure that the betweenvars and withinvars are factors
  factorvars <- vapply(data[, c(betweenvars, withinvars), drop=FALSE],
                       FUN=is.factor, FUN.VALUE=logical(1))
  
  if (!all(factorvars)) {
    nonfactorvars <- names(factorvars)[!factorvars]
    message("Automatically converting the following non-factors to factors: ",
            paste(nonfactorvars, collapse = ", "))
    data[nonfactorvars] <- lapply(data[nonfactorvars], factor)
  }
  
  # Get the means from the un-normed data
  datac <- summarySE(data, measurevar, groupvars=c(betweenvars, withinvars),
                     na.rm=na.rm, conf.interval=conf.interval, .drop=.drop)
  
  # Drop all the unused columns (these will be calculated with normed data)
  datac$sd <- NULL
  datac$se <- NULL
  datac$ci <- NULL
  
  # Norm each subject's data
  ndata <- normDataWithin(data, idvar, measurevar, betweenvars, na.rm, .drop=.drop)
  
  # This is the name of the new column
  measurevar_n <- paste(measurevar, "_norm", sep="")
  
  # Collapse the normed data - now we can treat between and within vars the same
  ndatac <- summarySE(ndata, measurevar_n, groupvars=c(betweenvars, withinvars),
                      na.rm=na.rm, conf.interval=conf.interval, .drop=.drop)
  
  # Apply correction from Morey (2008) to the standard error and confidence interval
  #  Get the product of the number of conditions of within-S variables
  nWithinGroups    <- prod(vapply(ndatac[,withinvars, drop=FALSE], FUN=nlevels,
                                  FUN.VALUE=numeric(1)))
  correctionFactor <- sqrt( nWithinGroups / (nWithinGroups-1) )
  
  # Apply the correction factor
  ndatac$sd <- ndatac$sd * correctionFactor
  ndatac$se <- ndatac$se * correctionFactor
  ndatac$ci <- ndatac$ci * correctionFactor
  
  # Combine the un-normed means with the normed results
  merge(datac, ndatac)
}

normDataWithin <- function(data=NULL, idvar, measurevar, betweenvars=NULL,
                           na.rm=FALSE, .drop=TRUE) {
  library(plyr)
  
  # Measure var on left, idvar + between vars on right of formula.
  data.subjMean <- ddply(data, c(idvar, betweenvars), .drop=.drop,
                         .fun = function(xx, col, na.rm) {
                           c(subjMean = mean(xx[,col], na.rm=na.rm))
                         },
                         measurevar,
                         na.rm
  )
  
  # Put the subject means with original data
  data <- merge(data, data.subjMean)
  
  # Get the normalized data in a new column
  measureNormedVar <- paste(measurevar, "_norm", sep="")
  data[,measureNormedVar] <- data[,measurevar] - data[,"subjMean"] +
    mean(data[,measurevar], na.rm=na.rm)
  
  # Remove this subject mean column
  data$subjMean <- NULL
  
  return(data)
}

#loading data sets: data=Experiment2, data1=Experiment1, data2=Experiment3, data 3=Experiment4
data <- read.csv(file.choose(), header=TRUE, fileEncoding="UTF-8-BOM")
data1 <- read.csv(file.choose(), header=TRUE, fileEncoding="UTF-8-BOM")
data2 <- read.csv(file.choose(), header=TRUE, fileEncoding="UTF-8-BOM")
data3 <- read.csv(file.choose(), header=TRUE, fileEncoding="UTF-8-BOM")

#participant exclusion
length(unique(data$Participant))
length(unique(data1$Participant))
length(unique(data2$Participant))
length(unique(data3$Participant))

data1=subset(data1, !data1$Participant%in%c("8a26c1e8a936b01376d0d52c692c9c67",
                                         "c84d34ee565c598d9fd817e3f6090e52"))

data2=subset(data2, !data2$Participant%in%c("1dd030bd145fd71c59f6e2377f6241de"))

#split up QUD conditions
data0=subset(data, !data$Condition%in%c("Estrong"))
data=subset(data, data$Condition%in%c("Estrong"))

data0$Item=as.integer(data0$Item)
data$Item=as.integer(data$Item)
data1$Item=as.integer(data1$Item)
data2$Item=as.integer(data2$Item)
data3$Item=as.integer(data3$Item)

###########VISUALIZATIONS###########
#make plot of all 4 experiments
datafull=rbind(data0, data1, data, data2, data3)

data_sum <- ddply(datafull, c("Item","Condition"), summarise,
                  N    = length(Response),
                  sum = sum(Response, na.rm=TRUE),
                  Percent = sum/N * 100)

data_sum$Condition=as.factor(data_sum$Condition)
levels(data_sum$Condition)
data_sum$Condition <- factor(data_sum$Condition, levels = c("ESI", "Eweak", "Estrong", "Eonly","Eonlystrong"))

data_sum$Scales <-c("allowed/obligatory","allowed/obligatory","allowed/obligatory","allowed/obligatory","allowed/obligatory",
                    "attractive/stunning","attractive/stunning","attractive/stunning","attractive/stunning","attractive/stunning",
                    "begin/complete","begin/complete","begin/complete","begin/complete","begin/complete",
                    "believe/know","believe/know","believe/know","believe/know","believe/know",
                    "big/enormous","big/enormous","big/enormous","big/enormous","big/enormous",
                    "cool/cold","cool/cold","cool/cold","cool/cold","cool/cold",
                    "damage/destroy","damage/destroy","damage/destroy","damage/destroy","damage/destroy",
                    "dark/black","dark/black","dark/black","dark/black","dark/black",
                    "difficult/impossible","difficult/impossible","difficult/impossible","difficult/impossible","difficult/impossible",
                    "dirty/filthy","dirty/filthy","dirty/filthy","dirty/filthy","dirty/filthy",
                    "dislike/loathe","dislike/loathe","dislike/loathe","dislike/loathe","dislike/loathe",
                    "double/triple","double/triple","double/triple","double/triple","double/triple",
                    "equally/more","equally/more","equally/more","equally/more","equally/more",
                    "funny/hilarious","funny/hilarious","funny/hilarious","funny/hilarious","funny/hilarious",
                    "good/excellent","good/excellent","good/excellent","good/excellent","good/excellent",
                    "happy/ecstatic","happy/ecstatic","happy/ecstatic","happy/ecstatic","happy/ecstatic",
                    "hard/unsolvable","hard/unsolvable","hard/unsolvable","hard/unsolvable","hard/unsolvable",
                    "harmful/deadly","harmful/deadly","harmful/deadly","harmful/deadly","harmful/deadly",
                    "here/everywhere","here/everywhere","here/everywhere","here/everywhere","here/everywhere",
                    "hungry/starving","hungry/starving","hungry/starving","hungry/starving","hungry/starving",
                    "intelligent/brilliant","intelligent/brilliant","intelligent/brilliant","intelligent/brilliant","intelligent/brilliant",
                    "intimidating/terrifying","intimidating/terrifying","intimidating/terrifying","intimidating/terrifying","intimidating/terrifying",
                    "largely/totally","largely/totally","largely/totally","largely/totally","largely/totally",
                    "like/love","like/love","like/love","like/love","like/love",
                    "match/exceed","match/exceed","match/exceed","match/exceed","match/exceed",
                    "mostly/entirely","mostly/entirely","mostly/entirely","mostly/entirely","mostly/entirely",
                    "old/ancient","old/ancient","old/ancient","old/ancient","old/ancient",
                    "once/twice","once/twice","once/twice","once/twice","once/twice",
                    "or/and","or/and","or/and","or/and","or/and",
                    "overweight/obese","overweight/obese","overweight/obese","overweight/obese","overweight/obese",
                    "overwhelmingly/unanimously","overwhelmingly/unanimously","overwhelmingly/unanimously","overwhelmingly/unanimously","overwhelmingly/unanimously",
                    "palatable/delicious","palatable/delicious","palatable/delicious","palatable/delicious","palatable/delicious",
                    "partially/completely","partially/completely","partially/completely","partially/completely","partially/completely",
                    "permit/require","permit/require","permit/require","permit/require","permit/require",
                    "polished/impeccable","polished/impeccable","polished/impeccable","polished/impeccable","polished/impeccable",
                    "possible/certain","possible/certain","possible/certain","possible/certain","possible/certain",
                    "pretty/beautiful","pretty/beautiful","pretty/beautiful","pretty/beautiful","pretty/beautiful",
                    "primarily/exclusively","primarily/exclusively","primarily/exclusively","primarily/exclusively","primarily/exclusively",
                    "probably/necessarily","probably/necessarily","probably/necessarily","probably/necessarily","probably/necessarily",
                    "reduce/eliminate","reduce/eliminate","reduce/eliminate","reduce/eliminate","reduce/eliminate",
                    "scared/petrified","scared/petrified","scared/petrified","scared/petrified","scared/petrified",
                    "serious/life-threatening","serious/life-threatening","serious/life-threatening","serious/life-threatening","serious/life-threatening",
                    "similar/identical","similar/identical","similar/identical","similar/identical","similar/identical",
                    "slow/stop","slow/stop","slow/stop","slow/stop","slow/stop",
                    "small/tiny","small/tiny","small/tiny","small/tiny","small/tiny",
                    "snug/tight","snug/tight","snug/tight","snug/tight","snug/tight",
                    "some/all","some/all","some/all","some/all","some/all",
                    "start/finish","start/finish","start/finish","start/finish","start/finish",
                    "survive/thrive","survive/thrive","survive/thrive","survive/thrive","survive/thrive",
                    "tired/exhausted","tired/exhausted","tired/exhausted","tired/exhausted","tired/exhausted",
                    "tolerate/encourage","tolerate/encourage","tolerate/encourage","tolerate/encourage","tolerate/encourage",
                    "try/succeed","try/succeed","try/succeed","try/succeed","try/succeed",
                    "ugly/hideous","ugly/hideous","ugly/hideous","ugly/hideous","ugly/hideous",
                    "understandable/articulate","understandable/articulate","understandable/articulate","understandable/articulate","understandable/articulate",
                    "unpleasant/disgusting","unpleasant/disgusting","unpleasant/disgusting","unpleasant/disgusting","unpleasant/disgusting",
                    "usually/always","usually/always","usually/always","usually/always","usually/always",
                    "want/need","want/need","want/need","want/need","want/need",
                    "warm/hot","warm/hot","warm/hot","warm/hot","warm/hot",
                    "well/superbly","well/superbly","well/superbly","well/superbly","well/superbly",
                    "willing/eager","willing/eager","willing/eager","willing/eager","willing/eager")

names <- c(
  `ESI` = "SI",
  `Eweak` = "Weak QUD",
  `Estrong` = "Strong QUD",
  `Eonly` = "Only",
  `Eonlystrong` = "QUD+only"
)

#ordering items according to SI rate in Experiment1
data_sum_3 <- dcast(data_sum, Scales ~ Condition, value.var="Percent")
data_sum_4 <- data_sum_3[order(data_sum_3$ESI),] 
data_sum_4$Scales = factor(data_sum_4$Scales, levels = data_sum_4$Scales)
data_sum_5 <- melt(data_sum_4, id.vars=c("Scales"))

ggplot(data_sum_5, aes(x=Scales, y=value)) + facet_wrap(~variable, ncol=5, labeller = as_labeller(names)) +
  geom_bar(position=position_dodge(), stat="identity") + theme_bw() + coord_flip() + 
  scale_fill_manual(values=cbbPalette) +
  theme(strip.text.x = element_text(size = 16), legend.position="right", axis.title.x = element_text(size=16),legend.title = element_text(size=12),
        legend.text = element_text(size = 12),
        axis.text.x  = element_text(size=12), axis.title.y = element_text(size=16),
        axis.text.y  = element_text(size=10), plot.title=element_text(size=12), text =  element_text(family="serif")) +
  xlab("Scales") + ylab("Percent of upper-bounded inference calculation") 

#make plots of subsets of experiments
data_sum_6=subset(data_sum_5, data_sum_5$variable%in%c("ESI","Eweak","Estrong"))

ggplot(data_sum_6, aes(x=Scales, y=value)) + facet_wrap(~variable, ncol=3, labeller = as_labeller(names)) +
  geom_bar(position=position_dodge(), stat="identity") + theme_bw() + coord_flip() + 
  scale_fill_manual(values=cbbPalette) + scale_y_continuous(limits=c(0, 100)) + 
  theme(strip.text.x = element_text(size = 16), legend.position="right", axis.title.x = element_text(size=16),legend.title = element_text(size=12),
        legend.text = element_text(size = 12),
        axis.text.x  = element_text(size=12), axis.title.y = element_text(size=16),
        axis.text.y  = element_text(size=10), plot.title=element_text(size=12), text =  element_text(family="serif")) +
  xlab("Scales") + ylab("Percent of SI calculation") 

data_sum_7=subset(data_sum_5, data_sum_5$variable%in%c("ESI","Eweak","Estrong","Eonly"))

ggplot(data_sum_7, aes(x=Scales, y=value)) + facet_wrap(~variable, ncol=4, labeller = as_labeller(names)) +
  geom_bar(position=position_dodge(), stat="identity") + theme_bw() + coord_flip() + 
  scale_fill_manual(values=cbbPalette) + scale_y_continuous(limits=c(0, 100)) + 
  theme(strip.text.x = element_text(size = 16), legend.position="right", axis.title.x = element_text(size=16),legend.title = element_text(size=12),
        legend.text = element_text(size = 12),
        axis.text.x  = element_text(size=12), axis.title.y = element_text(size=16),
        axis.text.y  = element_text(size=10), plot.title=element_text(size=12), text =  element_text(family="serif")) +
  xlab("Scales") + ylab("Percent of upper-bounded inference calculation") 

data_sum_8=subset(data_sum_5, data_sum_5$variable%in%c("ESI"))

ggplot(data_sum_8, aes(x=Scales, y=value)) +
  geom_bar(position=position_dodge(), stat="identity") + theme_bw() + coord_flip() + 
  scale_fill_manual(values=cbbPalette) + scale_y_continuous(limits=c(0, 100)) + 
  theme(strip.text.x = element_text(size = 16), legend.position="right", axis.title.x = element_text(size=16),legend.title = element_text(size=12),
        legend.text = element_text(size = 12),
        axis.text.x  = element_text(size=12), axis.title.y = element_text(size=16),
        axis.text.y  = element_text(size=11), plot.title=element_text(size=12), text =  element_text(family="serif")) +
  xlab("Scales") + ylab("Percent of SI calculation") 


###########STATISTICS###########
#Experiment 2: comparing strong vs. weak QUD
datastrongvsweak=rbind(data0,data)
datastrongvsweak$Condition=as.factor(datastrongvsweak$Condition)
levels(datastrongvsweak$Condition)
datastrongvsweak$Condition=relevel(datastrongvsweak$Condition, "Eweak")
contrasts(datastrongvsweak$Condition) <- c(-1/2, 1/2)
QUD_strongvsweak_model <- glmer(Response ~ Condition
                                + (1 |Participant)
                                + (1 + Condition |Item),
                                family= 'binomial',
                                data = datastrongvsweak)

summary(QUD_strongvsweak_model)

#comparing Experiment 1 (SI) vs 2 (QUD)
dataExp1vsExp2=rbind(data,data1,data0)
dataExp1vsExp2$Condition=as.factor(dataExp1vsExp2$Condition)
levels(dataExp1vsExp2$Condition)
contrasts(dataExp1vsExp2$Condition) <-contr.treatment(3,base=1)

QUD_Exp1vsExp2_model <- glmer(Response ~ Condition
                        + (1 |Participant)
                        + (1 + Condition |Item),
                        family= 'binomial',
                        data = dataExp1vsExp2)

summary(QUD_Exp1vsExp2_model)

#comparing Experiment 3 (only) vs 1 (SI)
dataSIvsstrongonly=rbind(data1,data2)
dataSIvsstrongonly$Condition=as.factor(dataSIvsstrongonly$Condition)
levels(dataSIvsstrongonly$Condition)
dataSIvsstrongonly$Condition=relevel(dataSIvsstrongonly$Condition, "ESI")
contrasts(dataSIvsstrongonly$Condition) <- c(-1/2, 1/2)

only_model <- glmer(Response ~ Condition
                   + (1 |Participant)
                   + (1 + Condition |Item),
                   family= 'binomial',
                   data = dataSIvsstrongonly)

summary(only_model)

#comparing Experiment 3 (only) vs strong QUD from Exp 2
datastrongQUDvsonly=rbind(data,data2)
datastrongQUDvsonly$Condition=as.factor(datastrongQUDvsonly$Condition)
levels(datastrongQUDvsonly$Condition)
datastrongQUDvsonly$Condition=relevel(datastrongQUDvsonly$Condition, "Estrong")
contrasts(datastrongQUDvsonly$Condition) <- c(-1/2, 1/2)

strongQUDvsonly_model <- glmer(Response ~ Condition
                    + (1 |Participant)
                    + (1 + Condition |Item),
                    family= 'binomial',
                    data = datastrongQUDvsonly)

summary(strongQUDvsonly_model)

#comparing Experiment 4 (QUD+only) vs 1 (SI)
dataSIvsstrongQonly=rbind(data1,data3)
dataSIvsstrongQonly$Condition=as.factor(dataSIvsstrongQonly$Condition)
levels(dataSIvsstrongQonly$Condition)
dataSIvsstrongQonly$Condition=relevel(dataSIvsstrongQonly$Condition, "ESI")
contrasts(dataSIvsstrongQonly$Condition) <- c(-1/2, 1/2)

Qonly_model <- glmer(Response ~ Condition
                    + (1 |Participant)
                    + (1 + Condition |Item),
                    family= 'binomial',
                    data = dataSIvsstrongQonly)

summary(Qonly_model)

#comparing Experiment 4 (QUD+only) vs Experiment 3 (only)
dataonlyvsstrongQonly=rbind(data2,data3)
dataonlyvsstrongQonly$Condition=as.factor(dataonlyvsstrongQonly$Condition)
levels(dataonlyvsstrongQonly$Condition)
contrasts(dataonlyvsstrongQonly$Condition) <- c(-1/2, 1/2)

onlyvsQonly_model <- glmer(Response ~ Condition
                     + (1 |Participant)
                     + (1 + Condition |Item),
                     family= 'binomial',
                     data = dataonlyvsstrongQonly)

summary(onlyvsQonly_model)


###########RELATIVE ENTROPY###########
px=data_sum_3$ESI/sum(data_sum_3$ESI)
sum(px*log(px*60,2))
#0.4659749

px_only=data_sum_3$Eonly/sum(data_sum_3$Eonly)
sum(px_only*log(px_only*60,2))
#0.04604028

px_onlyQUD=data_sum_3$Eonlystrong/sum(data_sum_3$Eonlystrong)
sum(px_onlyQUD*log(px_onlyQUD*60,2))
#0.006029651

px_QUD=data_sum_3$Estrong/sum(data_sum_3$Estrong)
sum(px_QUD*log(px_QUD*60,2))
#0.1225558

px_QUDweak=data_sum_3$Eweak/sum(data_sum_3$Eweak)
sum(px_QUDweak*log(px_QUDweak*60,2))
#0.3780404


###########RANK ORDER CORRELATIONS###########
#Experiment 2
cor.test(data_sum_3$Estrong, data_sum_3$Eweak, method=c("kendall"))
cor.test(data_sum_3$ESI, data_sum_3$Eweak, method=c("kendall"))
cor.test(data_sum_3$ESI, data_sum_3$Estrong, method=c("kendall"))

#Experiment 3
cor.test(data_sum_3$ESI, data_sum_3$Eonly, method=c("kendall"))
cor.test(data_sum_3$Eweak, data_sum_3$Eonly, method=c("kendall"))
cor.test(data_sum_3$Estrong, data_sum_3$Eonly, method=c("kendall"))

#Experiment 4
cor.test(data_sum_3$ESI, data_sum_3$Eonlystrong, method=c("kendall"))
cor.test(data_sum_3$Eweak, data_sum_3$Eonlystrong, method=c("kendall"))
cor.test(data_sum_3$Estrong, data_sum_3$Eonlystrong, method=c("kendall"))
cor.test(data_sum_3$Eonly, data_sum_3$Eonlystrong, method=c("kendall"))


###########Levene's test###########  
library(car)
#read in data_sum_levene.csv
levene <- read.csv(file.choose(), header=TRUE, fileEncoding="UTF-8-BOM")
levene$Condition=as.factor(levene$Condition) 

#compare everything with uniform
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("Uniform1","ESI"),])
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("Uniform1","Eweak"),])
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("Uniform1","Estrong"),])
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("Uniform1","Eonly"),])
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("Uniform2","Eonlystrong"),])

#Exp 1 vs strong QUD 
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("ESI","Estrong"),])

#Exp 1 vs weak QUD 
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("ESI","Eweak"),])

#Exp 1 vs only 
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("ESI","Eonly"),])

#strong QUD vs only 
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("Estrong","Eonly"),])

#Exp 4 vs Exp 1
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("ESI","Eonlystrong"),])

#Exp 4 vs only
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("Eonlystrong","Eonly"),])

#Exp 4 vs strong QUD
leveneTest(Percent ~ Condition, data = levene[levene$Condition%in%c("Eonlystrong","Estrong"),])
