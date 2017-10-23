#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <math.h>
#define NEG_PARAMATER -10000.0

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss(args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			property[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string name() const {
		auto it = property.find("name");
		return it != property.end() ? std::string(it->second) : "unknown";
	}
protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> property;
};

/**
 * evil (environment agent)
 * add a new random tile on board, or do nothing if the board is full
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public agent {
public:
	rndenv(const std::string& args = "") : agent("name=rndenv " + args) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
	}

	virtual action take_action(const board& after) {
		int space[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
		std::shuffle(space, space + 16, engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;
			std::uniform_int_distribution<int> popup(0, 9);
			int tile = popup(engine) ? 1 : 2;
			return action::place(tile, pos);
		}
		return action();
	}

private:
	std::default_random_engine engine;
};

/**
 * TODO: player (non-implement)
 * always return an illegal action
 */
class player : public agent {
public:
	player(const std::string& args = "") : agent("name=player " + args), alpha(0.0025f) {
		episode.reserve(32768);
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
		if (property.find("alpha") != property.end())
			alpha = float(property["alpha"]);

		if (property.find("load") != property.end())
			load_weights(property["load"]);
        else {
            //TODO: initialize the n-tuple network
            //initialize weight 
            int table_size = pow(_tile_total_state, _num_pt) ; 
            for ( int i =0 ; i < _num_tuple_set ;++i ){   
                weights.push_back( weight( table_size )  ) ;
            }  
            for ( int i =0 ; i < _num_tuple_set ;++i ){ 
                for ( int j =0 ; j < table_size ;++j ){ 
                    weights[i][j] = 0 ; 
                }  
            }     
        }
        // Initialize index base 
        for ( int i=0 ; i < _num_pt ; ++i ){
            _idx_base[i] = pow( _tile_total_state, i );
        }
        
        
	}
	~player() {
		if (property.find("save") != property.end())
			save_weights(property["save"]);
	}

	virtual void open_episode(const std::string& flag = "") {
		episode.clear();
		episode.reserve(32768*10);
	}

	virtual void close_episode(const std::string& flag = "") {
		// TODO: train the n-tuple network by TD(0)
        //std::cout << "Update Table " << std::endl ;
        if(alpha!=0){
            float update_score ; 
            for (int i=0 ; i <= int(episode.size()) ; ++i){
                if ( i == int(episode.size()) ){
                    update_score = - evaluate_board( episode[i-1].after, -1 ) ;
                    for (int j_num_tuple=0 ; j_num_tuple < _num_tuple ; ++j_num_tuple ){  
                        weights[ _tuple_set[j_num_tuple] ][ tuple_to_idx(episode[i-1].after, j_num_tuple) ]  += alpha * update_score ;               
                    } 
                }
                else{
                    update_score = episode[i].reward  + evaluate_board( episode[i].after, -1 ) - evaluate_board( episode[i].before, -1 ) ;
                    for (int j_num_tuple=0 ; j_num_tuple < _num_tuple ; ++j_num_tuple ){  
                        weights[ _tuple_set[j_num_tuple] ][ tuple_to_idx(episode[i].before, j_num_tuple) ]  += alpha * update_score ;               
                    } 
                }
            }
        }
	}

	virtual action take_action(const board& before) {
		////// TODO: select a proper action
		////// TODO: push the step into episode
        
        int opcode[] = { 0,1,2,3 };   
        
		//std::cout << "++++++++++++++++++++++++++++++++++++++++\n"<<"board:" << std::endl << before << std::endl ; 
        
        
        float estimated_evaluation_temp ;
        float estimated_evaluation = NEG_PARAMATER ; 
        float reward = 0.0 ; 
        float optimal_reward = 0.0 ; 
        int optimal_op = -1 ; 
        
		for (int op : opcode){
            board b_temp = before; 
            
			if (b_temp.move(op) != -1) {
                board b_temp = before;
                action move = action::move(op) ;
                reward = move.apply(b_temp)    ; 
                estimated_evaluation_temp = reward + evaluate_after_state_score(b_temp) ;
                
                //std::cout << "action: " << op << " with estimated reward " << reward << ", " <<estimated_evaluation_temp << std::endl ;
                
                if (estimated_evaluation < estimated_evaluation_temp){
                    estimated_evaluation = estimated_evaluation_temp ; 
                    optimal_op = op ;   
                    optimal_reward = float(reward) ;                     
                }
			}           
		}
        
       // std::cout << optimal_op << " is chosen \n" ;
        
        // complete last state logging
        if (_episode_temp.reward != NEG_PARAMATER){ 
            //std::cout << "test:" << _episode_temp.reward << std::endl ; 
            _episode_temp.after = before ; 
            episode.push_back(_episode_temp) ; 
            //std::cout << "push back into episode with episode size " << episode.size() << std::endl ; 
        }
        // New episode 
        _episode_temp.before = before ; 
        _episode_temp.move   = action::move(optimal_op) ; 
        _episode_temp.reward = optimal_reward ;
        

        if (optimal_op >= 0 )
            return action::move(optimal_op);
        else 
            return action();
        
	}
    int tuple_to_idx(const board& board_temp, int tuple_idx){
        int temp_idx = 0 ;
        for (int j_num_pt=0 ; j_num_pt < _num_pt ; ++j_num_pt){               
            temp_idx += board_temp[ _tuple_idx[tuple_idx][j_num_pt][0] ][ _tuple_idx[tuple_idx][j_num_pt][1] ] * _idx_base[j_num_pt] ;
        }
        return temp_idx ; 
    }
    float evaluate_board(const board& board_temp, int tuple_idx){
        /*
            index :
            -1 : evaluate the V(s)
            0-num_tuple : evaluate certain tuple 
        */
        float value = 0.0 ; 
        
        if ( tuple_idx == -1 ){
            for (int i_num_tuple=0 ; i_num_tuple < _num_tuple ; ++i_num_tuple){          
                value += weights[ _tuple_set[ i_num_tuple ] ][ tuple_to_idx(board_temp, i_num_tuple) ]   ;       
            }
            value = value/_num_tuple ; 
        }
        else if ( tuple_idx <= _num_tuple && tuple_idx >= 0) {
            value = weights[ _tuple_set[ tuple_idx ] ][ tuple_to_idx(board_temp, tuple_idx) ]   ;  
        }
        else {
            std::cerr << "No such operation" << std::endl ; 
        }
        return value ; 
    }
    float evaluate_after_state_score(const board& before){
        
        int space[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        float board_value = 0.0 ; 
        float num_empty_tile = 0.0 ; 
        int   tile = -1 ; 
        board b_after ; 
        action move ; 
        
		for (int pos : space) {
            
			if (before(pos) != 0) continue;
            
            b_after = before ; 
			tile = 1;
			move = action::place(tile, pos);
            move.apply(b_after) ;
            board_value += 0.9 * evaluate_board(b_after, -1);
            
            b_after = before ; 
			tile = 2;
			move = action::place(tile, pos);
            move.apply(b_after) ;
            board_value += 0.1 * evaluate_board(b_after, -1) ;  
            num_empty_tile += 1.0 ; 
            //std::cout << "score make : " << board_value << std::endl ; 
		}     
        board_value = board_value/num_empty_tile ; 
        //std::cout << "board_value : " <<board_value << std::endl ;  
		return board_value ;
    }
    
public:
	virtual void load_weights(const std::string& path) {
		std::ifstream in;
		in.open(path.c_str(), std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		size_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		weights.resize(size);
		for (weight& w : weights)
			in >> w;
		in.close();
	}

	virtual void save_weights(const std::string& path) {
		std::ofstream out;
		out.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		size_t size = weights.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : weights)
			out << w;
		out.flush();
		out.close();
	}
    

    
    
private:
    /*const int _num_tuple_set    = 2  ; 
    const int _num_tuple        = 8  ; 
    const int _num_pt           = 4  ; 
    const int _tile_total_state = 24 ; 
    float* _idx_base = new float[_num_pt] ; 
    const int _tuple_idx[8][4][2] ={
        { {0,0}, {0,1}, {0,2}, {0,3} },         
        { {0,0}, {1,0}, {2,0}, {3,0} },      
        { {3,0}, {3,1}, {3,2}, {3,3} },      
        { {0,3}, {1,3}, {2,3}, {3,3} },             
        { {1,0}, {1,1}, {1,2}, {1,3} },        
        { {2,0}, {2,1}, {2,2}, {2,3} },              
        { {0,1}, {1,1}, {2,1}, {3,1} },         
        { {0,2}, {1,2}, {2,2}, {3,2} } 

        
    };   
    const int _tuple_set[8] = { 0,0,0,0, 1,1,1,1 } ;*/
    
    const int _num_tuple_set    = 2  ; 
    const int _num_tuple        = 9  ; 
    const int _num_pt           = 4  ; 
    const int _tile_total_state = 24 ; 
    float* _idx_base = new float[_num_pt] ; 
    const int _tuple_idx[9][4][2] ={
        { {0,3}, {0,2}, {0,1}, {0,0} },         
        { {0,0}, {1,0}, {2,0}, {3,0} },      
        { {0,1}, {1,1}, {2,1}, {3,1} },      
        { {0,2}, {1,2}, {2,2}, {3,2} },             
        { {0,3}, {1,3}, {2,3}, {3,3} },  
        
        { {0,3}, {0,2}, {0,1}, {0,0} },              
        { {1,3}, {1,2}, {1,1}, {1,0} },         
        { {0,2}, {1,2}, {2,2}, {3,2} },
        { {0,3}, {1,3}, {2,3}, {3,3} }

        
    };   
    const int _tuple_set[9] = { 0,0,0,0,0, 1,1,1,1 } ;
    
    
    
    
    
    
    
    
    
    
    
    
        /*const int _tuple_idx[4][6][2] = {
        { {0,0},{1,0},{2,0},{3,0},{3,1},{2,1} },
        { {0,1},{1,1},{2,1},{3,1},{3,2},{2,2} },
        { {0,1},{1,1},{2,1},{2,2},{1,2},{0,2} },
        { {0,2},{1,2},{2,2},{2,3},{1,3},{0,3} }
    };
    const int _tuple_set[4] = {0,0,1,1};*/
    
    
    
    /*const int _tuple_idx[8][4][2] ={
        { {0,0}, {0,1}, {0,2}, {0,3} }, 
        { {0,0}, {1,0}, {2,0}, {3,0} },
        { {3,0}, {3,1}, {3,2}, {3,3} },
        { {0,3}, {1,3}, {2,3}, {3,3} },
        { {1,0}, {1,1}, {1,2}, {1,3} }, 
        { {2,0}, {2,1}, {2,2}, {2,3} },
        { {0,1}, {1,1}, {2,1}, {3,1} },        
        { {0,2}, {1,2}, {2,2}, {3,2} }
    };
    const int _tuple_set[8] = { 0,0,1,1, 1,1,0,0 } ;*/
    
    /*
    const int _tuple_idx[8][4][2] ={
        {{0,0},{0,1},{0,2},{0,3}},
        {{1,0},{1,1},{1,2},{1,3}},
        {{0,0},{1,0},{2,0},{3,0}},
        {{0,1},{1,1},{2,1},{3,1}},
        
        {{2,0},{2,1},{2,2},{2,3}},
        {{3,0},{3,1},{3,2},{3,3}},
        {{0,2},{1,2},{2,2},{3,2}},
        {{0,3},{1,3},{2,3},{3,3}},
    };  
    const int _tuple_set[8] = { 0,0,0,0, 1,1,1,1 } ;*/
    
    
    
	std::vector<weight> weights;
	struct state {
		// TODO: select the necessary components of a state
		board before;
		board after;
		action move;
		int reward = NEG_PARAMATER  ;
	};
    
	std::vector<state> episode;
    state _episode_temp ;
	float alpha;

private:
	std::default_random_engine engine;
};
