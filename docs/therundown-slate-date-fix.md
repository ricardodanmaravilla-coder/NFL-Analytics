# TheRundown slate-date matching fix

TheRundown can return an NFL weekly slate under the slate start date (commonly Thursday), while nflverse stores each game's actual kickoff date. Production now looks up the exact kickoff date first and then the four preceding dates, accepting a quote only when the normalized away/home pair matches exactly. This preserves the no-fabrication policy and allows Sunday/Monday games to match a Thursday slate snapshot.

Additional aliases cover common NFL abbreviations such as WSH/WAS, JAC/JAX, OAK/LV, SD/LAC, and STL/LA.
