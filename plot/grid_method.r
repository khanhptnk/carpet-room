
library(ggplot2)
library(reshape2)
library(ggpubr)


dat <- read.csv(file='grid_inun.csv')

dat <- melt(dat, id="Number.of.Updates")

print(dat)

p1 <- ggplot(data=dat, aes(x=Number.of.Updates, y=value, colour=variable)) +
    geom_line(size=1.5, alpha=0.9) + 
    scale_colour_manual(values=c("#D81B60", "#1E88E5", "#FFC107", "#004D40")) +
    #scale_fill_brewer(palette="Greens") +
    xlab("Training iterations \n (a) IntrUn") +
    ylim(0, 0.8) + 
    theme_bw() +
    theme(text=element_text(family="Palatino"),
          legend.position="none",                                                              
          legend.title=element_blank(),                                         
          legend.text=element_blank(),    
          axis.text.x = element_text(size=16),
          axis.text.y = element_text(size=16), 
          axis.title.x = element_text(size=23),
          axis.title.y = element_blank(),
          plot.margin = unit(c(.3,.3,0,0.3), "cm")
    )
                
dat <- read.csv(file='grid_exun.csv')

dat <- melt(dat, id="Number.of.Updates")

print(dat)

p2 <- ggplot(data=dat, aes(x=Number.of.Updates, y=value, colour=variable)) +
    geom_line(size=1.5, alpha=0.9) + 
    scale_colour_manual(values=c("#D81B60", "#1E88E5", "#FFC107", "#004D40")) +
    #scale_fill_brewer(palette="Greens") +
    xlab("Training iterations \n (b) ExtrUn") +
    ylim(0, 0.7) + 
    theme_bw() +
    theme(text=element_text(family="Palatino"),
          legend.position="none",                                                              
          legend.title=element_blank(),                                         
          legend.text=element_blank(),    
          axis.text.x = element_text(size=16),
          axis.text.y = element_text(size=16), 
          axis.title.x = element_text(size=23),
          axis.title.y = element_blank(),
          plot.margin = unit(c(.3,.3,0,0.3), "cm")
    )

dat <- read.csv(file='grid_apil.csv')

dat <- melt(dat, id="Number.of.Updates")

print(dat)

p3 <- ggplot(data=dat, aes(x=Number.of.Updates, y=value, colour=variable)) +
    geom_line(size=1.5, alpha=0.9) + 
    scale_colour_manual(values=c("#D81B60", "#1E88E5", "#FFC107", "#004D40")) +
    #scale_fill_brewer(palette="Greens") +
    xlab("Training iterations \n(c) APIL") +
    ylim(0, 0.5) + 
    theme_bw() +
    theme(text=element_text(family="Palatino"),
          legend.position="none",                                                              
          legend.title=element_blank(),                                         
          legend.text=element_blank(),    
          axis.text.x = element_text(size=16),
          axis.text.y = element_text(size=16), 
          axis.title.x = element_text(size=23),
          axis.title.y = element_blank(),
          plot.margin = unit(c(.3,.3,0,0.3), "cm")
    )

                
ggarrange(p1, p2, p3, ncol=3, nrow=1, heights=c(4, 4), align="hv", 
          common.legend = FALSE)


ggsave('grid_method.pdf', height=2.5, width=10)



