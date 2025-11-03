library(dplyr)
library(MASS)
library(ks)
library(slider)
library(ggplot2)
library(RColorBrewer)
library(ggnewscale)
library(cowplot)
library(HDInterval)
library(zoo)
library(tidyr)
library(tibble)
library(tseries)
library(forecast)
library(zoo)
library(patchwork)
library(jsonlite)
library(ggforce)
setwd("C:/Users/samil/Documents/wasp_MARL/outputs")
data_json = fromJSON('data.json')
decoys_df = data_json$decoys
schedule = data_json$schedule
n_cols = 5
new_schedule = data.frame()

for (i in seq_along(schedule)) {          # safe even when list is length 0
  elem <- schedule[[i]]                   # the i-th element (any type)
  df_elem = data.frame(matrix(elem,ncol = n_cols,byrow=TRUE))
  t = as.numeric(i)
  df_elem$time_step = t
  new_schedule = rbind(new_schedule,df_elem)
}
map = c(X1='id',X2='x',X3='y',X4='role',X5='found')
names(new_schedule)[match(names(map),names(new_schedule))] <- map
communicator_movement <-
  new_schedule %>%
  filter(time_step > 10, time_step < 200, role == 2) %>%
  mutate(
    bin_start = ((time_step - 1) %/% 10) * 10 + 1,
    bin_end   = bin_start + 9,
    ts_bin    = factor(
      paste0(bin_start, "–", bin_end),
      levels = paste0(seq(11, 191, by = 10), "–", seq(20, 200, by = 10))  # <- 10-step levels
    ),
    ord = -time_step   # <- ordering key (descending time)
  )
communicator_movement_trajectory_plot = ggplot(data=communicator_movement%>% filter(time_step<100 &time_step>10 & role==2))+geom_line(aes(x=time_step,y=(x**2+y**2)**(1/2),color=factor(id)))
communicator_movement_timeless_plot = ggplot(data=communicator_movement%>% filter(time_step<100 &time_step>95 & role==2))+geom_point(aes(x=x,y=y,color=factor(id)))+ ggforce::geom_circle(aes(x0=x,y0=y,r=1,fill=factor(id),color=factor(id)),alpha=0.1,inherit.aes = FALSE)+coord_fixed()
communicator_movement_long_timeless_plot = ggplot(data=communicator_movement%>% filter(time_step<100 &time_step>10 & role==2))+geom_path(aes(x=x,y=y,color=factor(id)))

explorer_movement <- new_schedule %>%
  filter(time_step > 10, role == 1)

rescuer_movement <- new_schedule %>%
  filter(time_step > 10, role == 3)

explorer_movement_timeless_plot = ggplot(data=new_schedule%>% filter(time_step<400 &time_step>10 & role==1))+geom_path(aes(x=x,y=y,color=factor(id)))+ ggforce::geom_circle(data=decoys_df,aes(x0=x,y0=y,r=6,fill=factor(role)),alpha=0.1,inherit.aes = FALSE)+coord_fixed()
id_explorers = new_schedule %>% filter(role==3)
id_explorers =unique(id_explorers$id)
rescuer_movement_timeless_plot = ggplot(data=new_schedule%>% filter(time_step>10 & id %in% id_explorers & role ==3))+geom_point(aes(x=x,y=y,color=factor(id)))+ ggforce::geom_circle(data=decoys_df%>%filter(role==4),aes(x0=x,y0=y,r=6,fill=factor(role)),alpha=0.1,inherit.aes = FALSE)+coord_fixed()
saturated_decoy = decoys_df%>%filter(role==4)

rescuer_df =  new_schedule %>%
  filter(time_step > 10, id %in% id_explorers, role == 3) %>%
  mutate(d2 = (x - saturated_decoy$x[1])^2 + (y - saturated_decoy$y[1])^2) 

rescuer_movement_trajectory_plot = ggplot(data=rescuer_df)+geom_line(aes(x=time_step,y=d2**(1/2)),color=factor(id))
