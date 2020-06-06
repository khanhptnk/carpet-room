library(ggplot2)

dat <- data.frame(
        object = factor(c("Intrinsic", "Intrinsic", "Intrinsic", "Intrinsic", 
                          "Extrinsic", "Extrinsic", "Extrinsic", "Extrinsic"),
                 levels=c("Intrinsic", "Extrinsic")),
        group = factor(c("Detm", "Rand", "TwoRand", "TwoDifDetm",
                         "Detm", "Rand", "TwoRand", "TwoDifDetm"), 
                levels=c("Detm", "Rand", "TwoRand", "TwoDifDetm")),
        value = c(0.03581, 0.71677, 0.71873, 0.05444,
                  0.00512, 0.00048, 0.00225, 0.55562)
)

print(dat)

p <- ggplot(data=dat, aes(x=object, y=value, fill=group)) +
    geom_bar(stat="identity", color="white", position=position_dodge(), width=0.8, size=1.5) + 
    geom_text(aes(label=sprintf("%0.2f", round(value, digits = 2))), 
              size=3, position=position_dodge(width=0.8), vjust=-.6) +
    ylim(0,0.8) + 
    scale_fill_manual(values=c("#D81B60", "#1E88E5", "#FFC107", "#004D40")) +
    #scale_fill_brewer(palette="Greens") +
    theme_bw() +
    theme(text=element_text(family="Palatino"),
          legend.position="top",                                                              
          legend.title=element_blank(),                                         
          legend.text=element_text(size=15),    
          axis.text.x = element_text(size=20),
          axis.text.y = element_text(size=20), 
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          plot.margin = unit(c(1,1.5,0.5,0.5), "cm")
          #panel.background = element_rect(fill = 'white', colour = 'black')
    )
                
         
ggsave('grid_uncertainty_grid.pdf', height=5, width=5)



