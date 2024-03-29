library(car)

remove.linearly.dependent.variables=function(data.frame){
	data.df=data.frame
	data.df.temp=data.df
	df.rank=qr(data.df)$rank
	rankifremoved=sapply(1:ncol(data.df), function (x) qr(data.df[,-x])$rank)
	linearly.dependent.var=colnames(data.df[which(rankifremoved == max(rankifremoved))])
	if (length(unique(rankifremoved))==1){
		print("No linearly dependent variables present")
		return(data.df.temp)
		}
	else{
		print(c("There is linear dependency among the following variables:", linearly.dependent.var))
	}
	while (length(linearly.dependent.var)>0 & length(unique(rankifremoved))>1){
		if (qr(data.df.temp[, !(names(data.df.temp) %in% rev(linearly.dependent.var)[1])])$rank==df.rank){
			data.df.temp=data.df.temp[, !(names(data.df.temp) %in% rev(linearly.dependent.var)[1])]
			}
		linearly.dependent.var=linearly.dependent.var[-length(linearly.dependent.var)]
		}
	print(c("The following variables have been removed:", names(data.df[which(!(names(data.df)) %in% names(data.df.temp))])))
	return(data.df.temp)
	}

remove.collinear.variables=function(data.df, threshold=20, bootstrap=100){		#Default set at 20 from ter Braak & Smilauer, 2002
	data.frame=remove.linearly.dependent.variables(data.df)	#Run vif() on data frame without singularity-causing variables
	data.expression="data=data.frame"
	if (nrow(data.frame)>ncol(data.frame)){
		iter=1
		collinearity.present=TRUE
		collinear.variables=NULL
			while (collinearity.present==TRUE){
				lhs.var=colnames(data.frame)[iter]
				rhs.var=colnames(data.frame)[(iter+1):ncol(data.frame)]
				index.rep=NULL
				data.frame.rep=rep("data.frame", length(rhs.var))
				for (i in 1:length(rhs.var)){
					index.rep[i]=paste("$", "'", as.character(rhs.var[i]), "'", sep="")
						}
				lm.rhs=paste(data.frame.rep, index.rep, sep="", collapse=" + ")
				lm.lhs=paste("data.frame$", "'", as.character(lhs.var), "'", sep="")
				lm.formula=paste(lm.lhs, lm.rhs, sep=" ~ ")
				lm.expression=paste("lm(", lm.formula, ", ", data.expression, ")", sep="")
				col.var=rhs.var[(vif(eval(parse(text=lm.expression)))>threshold)==TRUE]		#Logging variables strongly collinear with left hand-side variable
				if (length(col.var)>0){
					collinear.variables[[lhs.var]]=col.var		#Log collinear variables to list
					}
				data.frame=data.frame[, !(names(data.frame) %in% rhs.var[(vif(eval(parse(text=lm.expression)))>threshold)==TRUE])]
				iter=iter+1
				if (length(rhs.var)==2){
					collinearity.present=FALSE
					print(c("The collinear variables are: ", collinear.variables))
					print("End of script")
					}
				}
			return(data.frame)
			}


	}


Env.mat.no.collinearity = remove.collinear.variables(Env.mat.interpolated.NA)
#"There is linear dependency among the following variables:" "Zoopl.species"                                             "Copepods"                                                 
#"Cladocera"                                                
#"The following variables have been removed:" "Cladocera"                                 
#"The collinear variables are: "
#$Latitude
#"CO2"       "N2O"       "TOC"       "TN"        "TP"        "NO3"       "PO4"       "PON"       "POC"       "POP"       "Chla"      "a.phyto.m" "a.dom.m"   "tot.phyto" "C.P..at." 
#"C.P.calc"

colnames(Env.mat.no.collinearity)	#Get linearly independent variables
#"Latitude"      "Longitude"     "Altitude"      "Area"          "Depth"         "Temperature"   "Secchi"        "O2"            "CH4"           "pH"            "TIC"          
#"SiO2"          "APA"           "KdPAR"         "PPA"           "a.tripton.m"   "f.hetero"      "esd"           "car.chl"       "G"             "ZDW"           "Zoopl.species"
#"Copepods"      "Fish.species"

