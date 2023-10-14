## ACO Improvements

- [ ] Cambiare posizionamento stack con funzione di Davi -> anche la creazione degli stack?
- [X] Rimuovere deepcopy :(

## Column generation
### Model

- $c_n$ is the cost of the vehicle used in pattern $n$
- $w_{in}$ is the number of item of type $i$ in pattern $n$
- $N_i$ is the number of item of type $i$ that we have to transport.
- $x_n$ is the number of pattern $n$ used in the final solution.



$$
\min_{x} \quad \sum_{n \in \mathcal{N}} c_n x_n \\
\mathrm{s.t.} \quad  \sum_{n \in \mathcal{N}} w_{in} x_n \geq  N_i  \qquad \forall\ i\ \in \mathcal{I} \\
x_n \in \mathbb{Z}^+ \qquad \forall\ n\ \in \mathcal{N}.
$$

## Questions?
- Perchè il modello dovrebbe essere infeasible solo se la matrice W ha delle colonne a zero?  E come fa ad avere una colonna a zero? Significherebbe che ACO non ha messo nessun item nel veicolo. 
- Perchè il constrain >=? Non dovrebbe essere =?
- Il numero di variabili (*addVars()*) lo definiamo a priori?
- Le colonne vengono aggiunte dinamicamente?

## Ideas
- Cambiare *attractiveness* ad ogni iterazione considerando l'output precedente del modello.
