#.libPaths(c("M:/pc/Dokumenter/R/win-library", .libPaths()))
Env_data=read.table("C:/Users/laurenfo/Documents/COMSAT/COMSAT_environmental_metadata(spectral_data_lakes).tsv", stringsAsFactors=FALSE, header=T)
#Env.mat=scale(Env_data[,c(3:41, 43:44)], scale=T, center=T)
Env.mat=Env_data[,c(3:41, 43:44)]
#rownames(Env.mat) = rownames.OTU.matching.spectra
Taxonomy.table = readRDS("C:/Users/laurenfo/Documents/COMSAT/2019.01.31_Output_AE/tax_final.rds")	#Load Taxonomy table.
OTU.table = readRDS("C:/Users/laurenfo/Documents/COMSAT/2019.01.31_Output_AE/seqtab_nochim.rds")	#Load OTU table.
Spectra.table = read.table("C:/Users/laurenfo/Documents/COMSAT/Absorption spectra/COMSAT 2011 Absorption spectra DOM.txt", header=T, check.names=F)

rownames(OTU.table) = c(							#Change sequencing run IDs for lake ID and name in the OTU table.
"10000_Hurdalsjøen"  ,   "10001_Harestuvatnet"   , 
"170B_Gjersjøen"     ,   "170_Gjersjøen"         ,
"180_Øgderen"        ,   "189_Krøderen"          ,
"191_Rødbyvatnet"    ,  "194_Sperillen"          ,
"214_Gjesåssjøen"    , "2252_Rotnessjøen"        ,
"2268_Mylla"         ,     "233_Osensjøen"       ,   
"236_Rokossjøen"     ,   "2374_Klämmingen"       ,
"242_Sør Mesna"      ,   "252_Vermundsjøen"      ,
"261_Kalandsvatnet"  ,     "264_Myrkdalsvatnet"  ,   
"2678_Torrsjøn"      ,   "285_Rotevatnet"        ,
"2870_Visten"        ,     "2875_Näsrämmen"      ,
"2878_Rangsjön"      ,   "2887_Tisjön"           ,
"2888_Halsjøen"      ,   "288_Vatnevatnet"       , 
"2899_Jangen"        ,     "3017_Sör-älgen"      ,
"3019_Möckeln"       ,   "3020_Ljusnaren"        , 
"3025_Halvarsnoren"  ,     "3027_Nätsjön"        ,
"3029_Örlingen"      ,   "3031_Saxen"            , 
"3106_Långbjörken"   , "3160_Skattungen"         ,
"3165_Bäsingen"      ,   "3167_Tisken"           , 
"3185_Stora Almsjön" ,  "3189_Dragsjön"          ,
"3201_Milsjön"       ,   "3220_Stora Korslängen" ,
"326_Einavatnet"     ,     "328_Randsfjorden"    ,   
"3384_Hinsen"        ,    "3397_Storsjön"        ,
"3399_Grycken"       ,     "339_Ringsjøen"       ,
"340_Sæbufjorden"    ,   "344_Strondafjorden"    ,
"345_Trevatna"       ,    "349_Bogstadvannet"    ,
"3516_Holmsjön"      ,   "353_Aspern"            ,
"3541_Stornaggen"    ,     "361_Rødenessjøen"    ,
"363_Rømsjøen"       , "378_Hetlandsvatn"        ,
"380_Lutsivatn"      ,     "394_Vatsvatnet"      ,   
"395_Vostervatnet"   ,     "404_Jølstravatnet"   ,
"405_Oppstrynvatnet" ,     "433_Bandak"          , 
"436_Grungevatnet"   ,     "453_Vinjevatn"       ,
"481_Åsrumvatnet"    ,   "482_Bergsvannet"       ,
"486B_Goksjø"         ,  "486_Goksjø"            ,
"487_Hallevatnet"    ,     "498_Dagarn"          ,    
"5000_Forsjösjön"    , "519_Langen"
)

Spectra.table.2 = read.table("C:/Users/laurenfo/Documents/COMSAT/Absorption spectra/COMSAT 2011 Absorption spectra DOM ordered for 16S data.txt", header=T, check.names=F)
Spectra.mat = t(Spectra.table.2[,2:73])		#Select spectral data sites matching 16S data sites.
colnames(Spectra.mat) = Spectra.table.2[,1]
#Sample 353 from the OTU table has no corresponding spectral data.
OTU.table.spectral = OTU.table[c(1:7, 9:53, 55:74),]
OTU.table.spectral = OTU.table.spectral[,colSums(OTU.table.spectral)>0]	#Remove OTUs absent from site subset.
Taxonomy.table.spectral = Taxonomy.table[colnames(OTU.table.spectral),] #Remove OTUs absent from site subset.

rownames.OTU.matching.spectra = c(
"10000_Hurdalsjøen"  ,   "10001_Harestuvatnet"   , 
"170B_Gjersjøen"     ,   "170_Gjersjøen"         ,
"180_Øgderen"        ,   "189_Krøderen"          ,
"191_Rødbyvatnet"    ,
"214_Gjesåssjøen"    , "2252_Rotnessjøen"        ,
"2268_Mylla"         ,     "233_Osensjøen"       ,   
"236_Rokossjøen"     ,   "2374_Klämmingen"       ,
"242_Sør Mesna"      ,   "252_Vermundsjøen"      ,
"261_Kalandsvatnet"  ,     "264_Myrkdalsvatnet"  ,   
"2678_Torrsjøn"      ,   "285_Rotevatnet"        ,
"2870_Visten"        ,     "2875_Näsrämmen"      ,
"2878_Rangsjön"      ,   "2887_Tisjön"           ,
"2888_Halsjøen"      ,   "288_Vatnevatnet"       , 
"2899_Jangen"        ,     "3017_Sör-älgen"      ,
"3019_Möckeln"       ,   "3020_Ljusnaren"        , 
"3025_Halvarsnoren"  ,     "3027_Nätsjön"        ,
"3029_Örlingen"      ,   "3031_Saxen"            , 
"3106_Långbjörken"   , "3160_Skattungen"         ,
"3165_Bäsingen"      ,   "3167_Tisken"           , 
"3185_Stora Almsjön" ,  "3189_Dragsjön"          ,
"3201_Milsjön"       ,   "3220_Stora Korslängen" ,
"326_Einavatnet"     ,     "328_Randsfjorden"    ,   
"3384_Hinsen"        ,    "3397_Storsjön"        ,
"3399_Grycken"       ,     "339_Ringsjøen"       ,
"340_Sæbufjorden"    ,   "344_Strondafjorden"    ,
"345_Trevatna"       ,    "349_Bogstadvannet"    ,
"3516_Holmsjön"      ,
"3541_Stornaggen"    ,     "361_Rødenessjøen"    ,
"363_Rømsjøen"       , "378_Hetlandsvatn"        ,
"380_Lutsivatn"      ,     "394_Vatsvatnet"      ,   
"395_Vostervatnet"   ,     "404_Jølstravatnet"   ,
"405_Oppstrynvatnet" ,     "433_Bandak"          , 
"436_Grungevatnet"   ,     "453_Vinjevatn"       ,
"481_Åsrumvatnet"    ,   "482_Bergsvannet"       ,
"486B_Goksjø"         ,  "486_Goksjø"            ,
"487_Hallevatnet"    ,     "498_Dagarn"          ,    
"5000_Forsjösjön"    , "519_Langen"
)

rownames(Spectra.mat) = rownames.OTU.matching.spectra

#Interpolate missing values in environmental data by Multivariate Imputations by Chained Equations (MICE)
library(mice)
Env.mat.interpolated.NA=complete(mice(Env.mat))
rownames(Env.mat.interpolated.NA) = rownames.OTU.matching.spectra

#Env.mat.interpolated.NA and OTU.table.spectral are the matching environmental and community composition tables
#Env.mat.interpolated.NA[,28] is CDOM at 400 nm

CDOM.mat = matrix(Env.mat.interpolated.NA[,28], nrow(Env.mat.interpolated.NA),1)
colnames(CDOM.mat) = c("CDOM")
rownames(CDOM.mat) = rownames.OTU.matching.spectra
library(vegan)
ASV.hel = decostand(OTU.table.spectral, method="range")
#ASV.dist = vegdist(decostand(OTU.table.spectral, method="range"), method="bray")

pair.list = c()
CDOM.list = c()
sites.list = c()
for (site1 in 1:nrow(ASV.hel)){
		for (site2 in 1:nrow(ASV.hel)){
			pair = paste(min(site1, site2), max(site1, site2))
			if (site1!=site2 & !(pair %in% pair.list)){
				pair.list = c(pair.list, pair)
				ASV.dist = vegdist(ASV.hel[c(site1,site2),], method="bray")
				CDOM.x1 = min(CDOM.mat[site1], CDOM.mat[site2])
				CDOM.x2 = max(CDOM.mat[site1], CDOM.mat[site2])
				CDOM.mid = (CDOM.x1 + CDOM.x2)/2
				#print(c(rownames(ASV.hel)[site1], rownames(ASV.hel)[site2], ASV.dist, CDOM.x1, CDOM.x2, CDOM.mid))
				sites.list = c(sites.list, rownames(ASV.hel)[site1], rownames(ASV.hel)[site2])
				#print(c(CDOM.x1, CDOM.x2, CDOM.mid))
				CDOM.list = c(CDOM.list, CDOM.x1, CDOM.x2, CDOM.mid, ASV.dist)
				}
		
		}
	}
#CDOM.gradient.mat = cbind(matrix(sites.list,length(pair.list),2, byrow=T), matrix(CDOM.list,length(pair.list),3, byrow=T))
CDOM.sites.mat = matrix(sites.list,length(pair.list),2, byrow=T)
colnames(CDOM.sites.mat) = c("site_1", "site_2")
CDOM.gradient.mat =  matrix(CDOM.list,length(pair.list),4, byrow=T)
colnames(CDOM.gradient.mat) = c("CDOM.x1", "CDOM.x2", "CDOM.mid", "ASV.dist")

setwd("C:/Users/laurenfo/Documents/COMSAT")
write.table(CDOM.sites.mat, "CDOM.sites.mat.tsv", sep = "\t", row.names = FALSE)
write.table(CDOM.gradient.mat, "CDOM.gradient.mat.tsv", sep = "\t", row.names = FALSE)

qqnorm(sqrt(CDOM.gradient.mat[,3]))
qqline(sqrt(CDOM.gradient.mat[,3]))

qqnorm(log1p(CDOM.gradient.mat[,3]))
qqline(log1p(CDOM.gradient.mat[,3]))

qqnorm(CDOM.gradient.mat[,3])
qqline(CDOM.gradient.mat[,3])

# Generate meshgrid of distance values by CDOM gradient
pair.list = c()
ASV.dist.list = c()
for (site1 in 1:nrow(ASV.hel)){
		for (site2 in 1:nrow(ASV.hel)){
			pair = paste(rownames(ASV.hel)[site1], rownames(ASV.hel)[site2])
			pair.list = c(pair.list, pair)
			ASV.dist = vegdist(ASV.hel[c(site1,site2),], method="bray")
			ASV.dist.list = c(ASV.dist.list, ASV.dist)
		}
	}
CDOM.sites.mat.mesh = matrix(pair.list, nrow(ASV.hel), nrow(ASV.hel), byrow=T)
rownames(CDOM.sites.mat.mesh) = CDOM.mat
colnames(CDOM.sites.mat.mesh) = CDOM.mat
CDOM.sites.mat.mesh = CDOM.sites.mat.mesh[order(as.numeric(rownames(CDOM.sites.mat.mesh))), order(as.numeric(colnames(CDOM.sites.mat.mesh)))]

CDOM.gradient.mat.mesh =  matrix(ASV.dist.list, nrow(ASV.hel), nrow(ASV.hel), byrow=T)
rownames(CDOM.gradient.mat.mesh) = CDOM.mat
colnames(CDOM.gradient.mat.mesh) = CDOM.mat
CDOM.gradient.mat.mesh = CDOM.gradient.mat.mesh[order(as.numeric(rownames(CDOM.gradient.mat.mesh))), order(as.numeric(colnames(CDOM.gradient.mat.mesh)))]

diag.mesh = function(var.min, var.max, increment){	#Generate a molten diagonal matrix mesh for any interval
	gradient = seq(var.min, var.max, by=increment)
	pair.list = c()
	var.list = c()
	for (var.1 in 1:length(gradient)){
		for (var.2 in 1:length(gradient)){
			pair = paste(min(var.1, var.2), max(var.1, var.2))
			if (var.1!=var.2 & !(pair %in% pair.list)){
				pair.list = c(pair.list, pair)
				var.x1 = min(gradient[var.1], gradient[var.2])
				var.x2 = max(gradient[var.1], gradient[var.2])
				var.mid = (var.x2 + var.x1)/2
				var.list = c(var.list, var.x1, var.x2, var.mid)
				}
			}
		}
	var.gradient.mat =  matrix(var.list,length(pair.list),3, byrow=T)
	colnames(var.gradient.mat) = c("var.x1", "var.x2", "var.mid")
	return(var.gradient.mat)
	}

setwd("C:/Users/laurenfo/Documents/COMSAT")
write.table(CDOM.gradient.mat.mesh, "Bray_distances_by_CDOM_gradient_meshgrid.tsv", sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(CDOM.sites.mat.mesh, "CDOM.sites.meshgrid.tsv", sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(diag.mesh(0.21, 3.83, 0.01), "CDOM.diag.mesh.tsv", sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(sort(CDOM.mat), "CDOM.tsv", sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(OTU.table.spectral, "ASV_table.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(decostand(OTU.table.spectral, method="range"), "ASV_table_ranged.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA, "Metadata_table.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(scale(Env.mat.interpolated.NA, scale=T, center=T), "Metadata_table_scaled.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)

# Data subsets for Maxim
ASV_table_50_50 = OTU.table.spectral[,colSums(OTU.table.spectral==0)/nrow(OTU.table.spectral)==0.5]
ASV_table_50_50_binary = OTU.table.spectral[,colSums(OTU.table.spectral==0)/nrow(OTU.table.spectral)==0.5]
ASV_table_50_50_binary[ASV_table_50_50_binary>0]=1
ASV_table_min_10_observations = OTU.table.spectral[,colSums(OTU.table.spectral>0)>=10]

setwd("C:/Users/laurenfo/Documents/Courses/FYS-STK4155/Project 3")
write.table(Env.mat.interpolated.NA[,c("Area", "Depth", "Temperature", "Secchi", "O2", "CH4", "pH", "TIC", "SiO2", "KdPAR")], "Input_Metadata_10_variables.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(OTU.table.spectral, "Input_full_ASV_table.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(ASV_table_min_10_observations, "Input_ASV_table_min_10_observations.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(ASV_table_50_50_binary, "Output_ASV_table_50_50_binary.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("Area")], "Output_Metadata_Area.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("Depth")], "Output_Metadata_Depth.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("Temperature")], "Output_Metadata_Temperature.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("Secchi")], "Output_Metadata_Secchi.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("O2")], "Output_Metadata_O2.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("CH4")], "Output_Metadata_CH4.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("pH")], "Output_Metadata_pH.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("TIC")], "Output_Metadata_TIC.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("SiO2")], "Output_Metadata_SiO2.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(Env.mat.interpolated.NA[,c("KdPAR")], "Output_Metadata_KdPAR.tsv",  sep = "\t", row.names = FALSE, col.names = TRUE)

#Procrustes test to check for significant differences in clustering by XGboost and Random forests
XGboost.meta = read.table("C:/Users/laurenfo/Documents/Courses/FYS-STK4155/Project 3/Predicted_metadata_XGboost_subset_Maziar.tsv", header=T, check.names=F)
rownames(XGboost.meta) = rownames.OTU.matching.spectra
RF.meta = read.table("C:/Users/laurenfo/Documents/Courses/FYS-STK4155/Project 3/Predicted_metadata_RF_100_ASV_Maziar.tsv", 
					header=T, check.names=F)[,c("Latitude","Longitude","Altitude","Area","Depth","Temperature","Secchi","O2","CH4","pH","TIC","SiO2","KdPAR")]
rownames(RF.meta) = rownames.OTU.matching.spectra
Original.meta = Env.mat.interpolated.NA[,c("Latitude","Longitude","Altitude","Area","Depth","Temperature","Secchi","O2","CH4","pH","TIC","SiO2","KdPAR")]
rownames(Original.meta) = rownames.OTU.matching.spectra

library(vegan)
XGboost.dist=vegdist(scale(XGboost.meta), method="euclidean")	#Distance matrix for XGboost-predicted data, Q type analysis
RF.dist=vegdist(scale(RF.meta), method="euclidean")	#Distance matrix for RF-predicted data, Q type analysis
Original.dist=vegdist(scale(Original.meta), method="euclidean")	#Distance matrix for original metadata, Q type analysis

XGboost.protest=protest(XGboost.dist, Original.dist, scores = "sites", permutations = how(nperm = 999))	#Pairwise comparison of distance matrices for XGboost and original metadata
RF.protest=protest(RF.dist, Original.dist, scores = "sites", permutations = how(nperm = 999))	#Pairwise comparison of distance matrices for Random Forest and original metadata
XGboost.RF.protest=protest(XGboost.dist, RF.dist, scores = "sites", permutations = how(nperm = 999))	#Pairwise comparison of distance matrices for XGboost and Random Forest

library(dendextend)
XGboost.dendro = as.dendrogram(hclust(XGboost.dist, method="average"))
RF.dendro = as.dendrogram(hclust(RF.dist, method="average"))
Original.dendro = as.dendrogram(hclust(Original.dist, method="average"))
dl1 = dendlist(XGboost.dendro, Original.dendro)
dl2 = dendlist(RF.dendro, Original.dendro)
dl3 = dendlist(XGboost.dendro, RF.dendro)
setwd("C:/Users/laurenfo/Documents/Courses/FYS-STK4155/Project 3")
pdf("Metadata_predictions_tanglegrams.pdf")
tanglegram(dl1, sort = T, common_subtrees_color_lines = F, highlight_distinct_edges = F, 
highlight_branches_lwd = F, main_left = "XGboost", main_right = "True data", 
sub=paste("Procrustes test", "\n", "Correlation:", round(XGboost.protest[[6]], 4), "\n", "Significance:", XGboost.protest[13]),
common_subtrees_color_branches = FALSE, margin_inner = 7, lwd = 0.5, lab.cex = 0.8, cex_sub=0.8, axes=F)
tanglegram(dl2, sort = T, common_subtrees_color_lines = F, highlight_distinct_edges = F, 
highlight_branches_lwd = F, main_left = "Random Forest", main_right = "True data", 
sub=paste("Procrustes test", "\n", "Correlation:", round(RF.protest[[6]], 4), "\n", "Significance:", RF.protest[13]),
common_subtrees_color_branches = FALSE, margin_inner = 7, lwd = 0.5, lab.cex = 0.8, cex_sub=0.8, axes=F)
tanglegram(dl3, sort = T, common_subtrees_color_lines = F, highlight_distinct_edges = F, 
highlight_branches_lwd = F, main_left = "XGboost", main_right = "Random Forest", 
sub=paste("Procrustes test", "\n", "Correlation:", round(XGboost.RF.protest[[6]], 4), "\n", "Significance:", XGboost.RF.protest[13]),
common_subtrees_color_branches = FALSE, margin_inner = 7, lwd = 0.5, lab.cex = 0.8, cex_sub=0.8, axes=F)
dev.off()


# Partial out from ASV table variables uncorrelated with CDOM
Env.scaled = scale(Env.mat.interpolated.NA, scale=T, center=T)

reorder_cor.mat = function(cor.mat){
	# Use correlation between variables as distance
	dd = as.dist((1-cor.mat)/2)
	hc = hclust(dd)
	cor.mat = cor.mat[hc$order, hc$order]
	return(cor.mat)
	}

library(reshape2)

get_lower_triangle = function(cor.mat){
    cor.mat[upper.tri(cor.mat)] <- NA
    return(cor.mat)
  }
get_upper_triangle = function(cor.mat){
    cor.mat[lower.tri(cor.mat)]<- NA
    return(cor.mat)
  }

cor_p = function(env.mat){
	env.cor = melt(get_lower_triangle(reorder_cor.mat(cor(env.mat, use="all.obs", method="pearson"))), na.rm = TRUE)
	env.cor.p = env.cor
	for (i in 1:nrow(env.cor)){
		v1 = as.character(env.cor[i, 1])
		v2 = as.character(env.cor[i, 2])
		env.cor.p[i,3] = p.adjust(cor.test(env.mat[,v1],env.mat[,v2])$p.value, method = "holm", n = length(env.cor$value))
		}
	return(list(env.cor, env.cor.p))
	}

env.mat.list = cor_p(Env.scaled)
env.cor = env.mat.list[[1]]
env.cor.p = env.mat.list[[2]]

# Heatmap
library(ggplot2)
pdf("Environmental_variables_correlation_matrix.pdf")
ggplot(data = env.cor, aes(Var2, Var1, fill = value))+
 geom_tile(color = "white")+
 scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
   midpoint = 0, limit = c(-1,1), space = "Lab", 
   name="Pearson\nCorrelation") +
  theme_minimal()+ 
 scale_x_discrete(position = "top") +
 theme(axis.text.x = element_text(angle = 45, vjust = 1, 
    size = 8, hjust = 0))+
 coord_fixed()+
 # P values for correlations are shown as text in tiles
 geom_text(data = env.cor.p, aes(Var2, Var1, label = round(value, 2)), color = "black", size = 1.5) +
theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  panel.grid.major = element_blank(),
  panel.border = element_blank(),
  panel.background = element_blank(),
  axis.ticks = element_blank(),
  legend.justification = c(1, 0),
  legend.position = c(0.6, 0.1),
  legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                title.position = "top", title.hjust = 0.5))
dev.off()

env.cor.p[env.cor.p$Var1=="a.dom.m"|env.cor.p$Var2=="a.dom.m",]	#p-values for correlations with CDOM
env.cor.p[env.cor.p$Var1=="a.dom.m" & env.cor.p$value<0.05|env.cor.p$Var2=="a.dom.m" & env.cor.p$value<0.05,]	#p-values for significant correlations with CDOM
env.cor.p[env.cor.p$value<0.05 & env.cor.p$Var1!=env.cor.p$Var2,]	#Correlations and p-values for significantly correlated variables

network_df = function(env.mat, p.value = 0.05){
	env.cor.out = cor_p(env.mat)
	env.cor = env.cor.out[[1]]
	env.cor.p = env.cor.out[[2]]
	raw.df = env.cor.p[env.cor.p$value<p.value & env.cor.p$Var1!=env.cor.p$Var2,]
	node.list = as.data.frame(cbind(matrix(1:length(colnames(env.mat)),length(colnames(env.mat)),1), matrix(colnames(env.mat),length(colnames(env.mat)),1)))
	node.list$V1 = 1:nrow(node.list)
	node.list$V2 = as.character(node.list$V2)
	network.p.df = raw.df
	network.p.df$Var1 = 1:nrow(raw.df)
	network.p.df$Var2 = 1:nrow(raw.df)
	network.cor.df = network.p.df
	for (i in 1:nrow(raw.df)){
		network.p.df[i,1] = as.integer(node.list[node.list$V2==as.character(raw.df[i, 1]), 1])
		network.p.df[i,2] = as.integer(node.list[node.list$V2==as.character(raw.df[i, 2]), 1])
		network.p.df[i,3] = as.numeric(-log(raw.df[i, 3]))
		network.cor.df[i,1] = network.p.df[i,1]
		network.cor.df[i,2] = network.p.df[i,2]
		network.cor.df[i,3] = abs(env.cor[env.cor$Var1==as.character(raw.df$Var1[i]) & env.cor$Var2==as.character(raw.df$Var2[i]), 3])
		}
	return(list(network.cor.df, network.p.df, node.list))
	}

#http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization
#https://briatte.github.io/ggnet/
library(network)
library(sna)
library(ggplot2)
library(GGally)
pdf("Metadata_networks.pdf")
network_df.out = network_df(Env.scaled, 0.05)
env.network_cor_df = network_df.out[[1]]
env.network_p_df = network_df.out[[2]]
env.node_list = network_df.out[[3]]
env.network = network(env.network_cor_df, vertex.attr = env.node_list, matrix.type = "edgelist", directed = FALSE)
ggnet2(env.network, size = "degree", label=env.node_list$V2, label.size=2, edge.label.size = 0.1, edge.size = 0.01)+
ggtitle(label="p=0.05")

network_df.out = network_df(Env.scaled, 5e-4)
env.network_cor_df = network_df.out[[1]]
env.network_p_df = network_df.out[[2]]
env.node_list = network_df.out[[3]]
env.network = network(env.network_cor_df, vertex.attr = env.node_list, matrix.type = "edgelist", directed = FALSE)
ggnet2(env.network, size = "degree", label=env.node_list$V2, label.size=2, edge.label.size = 0.1, edge.size = 0.01)+
ggtitle(label="p=0.0005")
dev.off()

# Based on significance of correlations, Secchi, O2, Altitude, f.hetero, SiO2, CO2, TOC, a.dom.m, KdPAR, a.tripton.m
# must not be partialled out. Otherwise, information in the CDOM (a.dom.m) gradient would be lost.
Env.mat.interpolated.NA[,c(7, 8, 3, 31, 18, 9, 14, 28, 25, 29)]
Env.scaled[,c()]

ASV.cca=cca(X=ASV.hel, Y=Env.scaled[,c(7, 8, 3, 31, 18, 9, 14, 28, 25, 29)], 
Z=Env.scaled[,!(1:ncol(Env.scaled) %in% c(7, 8, 3, 31, 18, 9, 14, 28, 25, 29))])	#Y is the CDOM data selected as explanatory to X, Z are the environmental variables to be partialled out.
anova(ASV.cca)
RsquareAdj(ASV.cca)
  
partial.r(data, x=Env.scaled[,28], y=,use="pairwise",method="pearson")

Env.scaled[,!(1:ncol(Env.scaled) %in% c(7, 8, 3, 31, 18, 9, 14, 28, 25, 29))]