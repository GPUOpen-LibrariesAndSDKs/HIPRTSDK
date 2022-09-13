#include <tutorials/common/HipTestBase.h>
#include <cstdlib>
#include <numeric>

class Test : public HipTestBase
{
  public:
	void run() 
	{
		hiprtContext ctxt;
		hiprtError e = hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, &ctxt );
		ASSERT( e == hiprtSuccess );
		printf( "Create hip-hiprt context\n" );
	}
};

int main( int argc, char** argv )
{
	Test test;
	test.init( 0 );
	test.run();

	return 0;
}


