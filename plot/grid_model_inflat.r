
library(ggplot2)
library(reshape2)
library(ggpubr)


dat <- read.csv(file='grid_model_inflat.csv')

dat <- melt(dat, id="Number.of.Updates")

print(dat)

p1 <- ggplot(data=dat, aes(x=Number.of.Updates, y=value, colour=variable)) +
    geom_line(size=1.5, alpha=0.9) + 
    scale_colour_manual(
          name="#samples",
          labels=c("10", "20", "50", "100", "500"),
          #values=c("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#DDF74B")) +
          values=c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442")) +

    #scale_fill_brewer(palette="Greens") +
    xlab("Training iterations") + 
    ylab("Approx. model uncertainty") +
    #ylim(0, 0.8) + 
    theme_bw() +
    theme(text=element_text(family="Palatino"),
          legend.position="top",                                                              
          legend.title=element_text(size=20),                                         
          legend.text=element_text(size=15),    
          axis.text.x = element_text(size=20),
          axis.text.y = element_text(size=20), 
          axis.title.x = element_text(size=23),
          axis.title.y = element_text(size=20),
          plot.margin = unit(c(.5,.5,.5,.5), "cm")
    )
                

ggsave('grid_model_inflat.pdf', height=4, width=6)



