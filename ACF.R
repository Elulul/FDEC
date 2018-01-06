df <- read.csv2("C:/Users/lulu/Documents/Projet FDEC/donnéesNetoyés.csv",sep = ',',colClasses=c(rep("factor",24)))


library(FactoMineR)



selec <- subset(df, select=c(1:17))
# selec <- data.frame(selec,defaut = df[[20]])

selec <- selec[,-2]

selec <- selec[,-2]
selec <- selec[,-2]
selec <- selec[,-2]
selec <- selec[,-2]
selec <- selec[,-2]
selec <- selec[,-2]
selec <- selec[,-2]
selec <- selec[,-1]





selec2 <- head(selec,1000)



res <- MFA(selec, group = c(rep(1,8)) , type = c(rep("n",8)) , graph = TRUE)

plot(res)


res <- MCA(selec)